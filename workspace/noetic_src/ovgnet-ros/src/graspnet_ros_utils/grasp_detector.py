import numpy as np
import open3d as o3d
import open3d_plus as o3dp
from scipy.spatial.transform import Rotation as R
import copy
import os
import sys
import rospkg

rospack = rospkg.RosPack()

GRASPNET_ROS_DIR = rospack.get_path('graspnet-ros')
sys.path.append(os.path.join(GRASPNET_ROS_DIR, 'src'))

from graspnet_baseline import GraspNetBaseLine

# Modified from https://github.com/OCRTOC/OCRTOC_software_package/blob/master/ocrtoc_perception/src/ocrtoc_perception/perceptor.py
class Graspnet:
    def __init__(self, checkpoint_path, refine_approach_dist, dist_thresh, angle_thresh, mask_thresh):
        self.checkpoint_path = checkpoint_path
        self.refine_approach_dist = refine_approach_dist
        self.dist_thresh = dist_thresh
        self.angle_thresh = angle_thresh
        self.mask_thresh = mask_thresh
        self.graspnet_baseline = GraspNetBaseLine(checkpoint_path = self.checkpoint_path)

    def compute_grasp_pose(self, full_pcd):
        points, _ = o3dp.pcd2array(full_pcd)
        grasp_pcd = copy.deepcopy(full_pcd)
        grasp_pcd.points = o3d.utility.Vector3dVector(-points)

        # generating grasp poses.
        gg = self.graspnet_baseline.inference(grasp_pcd)
        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices
        gg.translations = gg.translations + gg.rotation_matrices[:, :, 0] * self.refine_approach_dist
        gg = self.graspnet_baseline.collision_detection(gg, points)

        # all the returned result in 'world' frame. 'gg' using 'graspnet' gripper frame.
        return gg

    def assign_grasp_pose(self, gg, object_poses):
        grasp_poses = dict()
        grasp_pose_set = []
        dist_thresh = self.dist_thresh
        # - dist_thresh: float of the minimum distance from the grasp pose center to the object center. The unit is millimeter.
        angle_thresh = self.angle_thresh
        # - angle_thresh:
        #             /|
        #            / |
        #           /--|
        #          /   |
        #         /    |
        # Angle should be smaller than this angle

        # gg: GraspGroup in 'world' frame of 'graspnet' gripper frame.
        # x is the approaching direction.
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        scores = gg.scores

        # move the center to the eelink frame
        # Note that here is rs[:,:,0] before
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape = (len(rs), 3, 3), dtype = np.float32)

        # the coordinate systems are different in graspnet and ocrtoc
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]

        # min_dist: np.array of the minimum distance to any object(must > dist_thresh)
        min_dists = np.inf * np.ones((len(rs)))

        # min_object_ids: np.array of the id of the nearest object.
        min_object_ids = -1 * np.ones(shape = (len(rs)), dtype = np.int32)

        # first round to find the object that each grasp belongs to.
        angle_mask = (rs[:, 2, 0] < -np.cos(angle_thresh / 180.0 * np.pi))
        for i, object_name in enumerate(object_poses.keys()):
            object_pose = object_poses[object_name]

            dists = np.linalg.norm(ts - object_pose[:3,3], axis=1)
            object_mask = np.logical_and(dists < min_dists, dists < dist_thresh)

            min_object_ids[object_mask] = i
            min_dists[object_mask] = dists[object_mask]
        remain_gg = []
        
        # second round to calculate the parameters
        for i, object_name in enumerate(object_poses.keys()):
            obj_id_mask = (min_object_ids == i)
            add_angle_mask = (obj_id_mask & angle_mask)
            # For safety and planning difficulty reason, grasp pose with small angle with gravity direction will be accept.
            # if no grasp pose is available within the safe cone. grasp pose with the smallest angle will be used without
            # considering the angle.
            if np.sum(add_angle_mask) < self.angle_thresh: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                mask = obj_id_mask
                sorting_method = 'angle'
            else:
                mask = add_angle_mask
                sorting_method = 'score'
            # print(f'{object_name} using sorting method: {sorting_method}, mask num: {np.sum(mask)}')
            i_scores = scores[mask]
            i_ts = ts[mask]
            i_eelink_rs = eelink_rs[mask]
            i_rs = rs[mask]
            i_gg = gg[mask]

            if np.sum(mask) < self.mask_thresh: # actually this should be mask == 0, for safety reason, < 0.5 is used.
                # ungraspable
                grasp_poses[object_name] = None
            else:
                grasp_poses[object_name] = []
                for i in range(len(i_gg)):
                    remain_gg.append(i_gg[i].to_open3d_geometry())
                    grasp_rotation_matrix = i_eelink_rs[i]
                    if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                        # print('\033[031mLeft Hand Coordinate System Grasp!\033[0m')
                        grasp_rotation_matrix[:,0] = - grasp_rotation_matrix[:, 0]
                    # else:
                    #     print('\033[032mRight Hand Coordinate System Grasp!\033[0m')
                    
                    grasp_pose = np.zeros(7)
                    grasp_pose[:3] = [i_ts[i][0], i_ts[i][1], i_ts[i][2]]
                    r = R.from_matrix(grasp_rotation_matrix)
                    grasp_pose[-4:] = r.as_quat()

                    grasp_poses[object_name].append(grasp_pose)
                    grasp_pose_set.append(grasp_pose)
                    
        return grasp_pose_set, grasp_poses, remain_gg

    def filter_grasps_by_score_and_angle(self, gg, min_score=0.15):
        """Filter grasps by score and angle with gravity"""
        # Filter by score
        del_index = []
        for index, value in enumerate(gg):
            if value.score < min_score:
                del_index.append(index)
        
        # Remove low-score grasps
        for i in reversed(del_index):
            gg.remove(i)
        
        # Get translations, rotations, and scores
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        scores = gg.scores
        
        # Move centers to the eelink frame
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        
        # Convert to eelink rotation matrices
        eelink_rs = np.zeros(shape=(len(rs), 3, 3), dtype=np.float32)
        eelink_rs[:,:,0] = rs[:,:,2]   # Approach direction 
        eelink_rs[:,:,1] = -rs[:,:,1]  # Gripper y-axis
        eelink_rs[:,:,2] = rs[:,:,0]   # Gripper z-axis
        
        # Filter by angle with gravity (z-axis)
        # Approach vector should be pointing downward
        angle_mask = (rs[:, 2, 0] < -np.cos(self.angle_thresh / 180.0 * np.pi))
        
        # If too few grasps pass the angle filter, use all grasps sorted by angle
        if np.sum(angle_mask) < self.mask_thresh:
            # Calculate angle with gravity for all grasps
            angles = np.arccos(-rs[:, 2, 0]) * 180.0 / np.pi
            # Sort by angle
            sorted_indices = np.argsort(angles)
            filtered_gg = gg[sorted_indices[:20]]  # Take top 20 grasps with best angle
            return filtered_gg, eelink_rs[sorted_indices[:20]]
        else:
            # Use grasps that pass angle filter, sort by score
            filtered_gg = gg[angle_mask]
            sorted_indices = np.argsort(-scores[angle_mask])  # Negative for descending
            filtered_gg = filtered_gg[sorted_indices[:20]]  # Take top 20 grasps with best scores
            return filtered_gg, eelink_rs[angle_mask][sorted_indices[:20]]

    def convert_grasps_to_poses(self, gg, eelink_rs):
        """Convert GraspGroup to a list of grasp poses (position + quaternion)"""
        grasp_poses = []
        geometries = []
        scores = []
        
        for i in range(len(gg)):
            # Add visualization geometry
            geometries.append(gg[i].to_open3d_geometry())

            scores.append(gg[i].score)
            
            # Get grasp rotation matrix
            grasp_rotation_matrix = eelink_rs[i]
            
            # Ensure right-hand coordinate system
            if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                grasp_rotation_matrix[:,0] = -grasp_rotation_matrix[:, 0]
            
            # Convert to position + quaternion format
            grasp_pose = np.zeros(7)
            grasp_pose[:3] = gg[i].translations + gg[i].rotation_matrices[:,:,0] * gg[i].depths  # Position
            r = R.from_matrix(grasp_rotation_matrix)
            grasp_pose[3:] = r.as_quat()  # Quaternion
            
            grasp_poses.append(grasp_pose)
        
        return grasp_poses, geometries, scores
    
    def grasp_detection_real_world(self, full_pcd, min_score=0.15):
        """
        Generate grasping poses for a real-world setting without known object poses.
        
        Args:
            full_pcd: Open3D point cloud of the scene
            min_score: Minimum score threshold for grasp filtering
            
        Returns:
            list, list: List of grasp poses (position + quaternion), and visualization geometries
        """
        try:
            # Check if the point cloud is valid
            if full_pcd is None or len(full_pcd.points) < 10:
                rospy.logwarn("Invalid or empty point cloud for grasp detection")
                return [], []
            
            # Compute grasp candidates
            rospy.loginfo(f"Computing grasp poses on point cloud with {len(full_pcd.points)} points")
            gg = self.compute_grasp_pose(full_pcd)
            
            # Log number of grasps found
            rospy.loginfo(f"Found {len(gg)} grasp candidates before filtering")
            
            if len(gg) == 0:
                rospy.logwarn("No grasp candidates found")
                return [], []
            
            # Filter grasps by score and angle
            filtered_gg, eelink_rs = self.filter_grasps_by_score_and_angle(gg, min_score)
            
            # Log number of grasps after filtering
            rospy.loginfo(f"Filtered to {len(filtered_gg)} grasp candidates")
            
            if len(filtered_gg) == 0:
                rospy.logwarn("No grasp candidates remain after filtering")
                return [], []
            
            # Convert to grasp poses
            grasp_poses, geometries, scores = self.convert_grasps_to_poses(filtered_gg, eelink_rs)
            
            # Optional visualization
            if False:  # Set to True for debugging
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([frame, full_pcd] + geometries)
                
            return grasp_poses, geometries, scores
            
        except Exception as e:
            rospy.logerr(f"Error in grasp detection: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return [], []

    # def  grasp_detection(self, full_pcd, object_poses=None):
    #     '''
    #     Generate object 6d poses and grasping poses.
    #     Only geometry infomation is used in this implementation.

    #     There are mainly three steps.
    #     - Moving the camera to different predefined locations and capture RGBD images. Reconstruct the 3D scene.
    #     - Generating objects 6d poses by mainly icp matching.
    #     - Generating grasping poses by graspnet-baseline.

    #     Args:
    #         object_list(list): strings of object names.
    #         pose_method: string of the 6d pose estimation method, "icp" or "superglue".
    #     Returns:
    #         dict, dict: object 6d poses and grasp poses.
    #     '''

    #     # generate grasping poses in a scene
    #     gg = self.compute_grasp_pose(full_pcd)
    #     del_index = []
    #     for index, value in enumerate(gg):
    #         if value.score < 0.15:
    #             del_index.append(index)
    #     for i in reversed(del_index):
    #         gg.remove(i)

    #     grasp_pose_set, grasp_pose_dict, remain_gg = self.assign_grasp_pose(gg, object_poses)
        
    #     # visualization
    #     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    #     # o3d.visualization.draw_geometries([frame, full_pcd, *gg.to_open3d_geometry_list()])

    #     return grasp_pose_set, grasp_pose_dict, remain_gg
    def grasp_detection(self, full_pcd, object_poses=None):
        '''
        Generate object 6d poses and grasping poses.
        Only geometry information is used in this implementation.
        
        Args:
            full_pcd: Open3D point cloud
            object_poses: Dictionary of object poses (optional, for simulation)
            
        Returns:
            dict, dict: object 6d poses and grasp poses.
        '''
        # If no object poses are provided, use the real-world version
        if object_poses is None or len(object_poses) == 0:
            grasp_poses, geometries, scores = self.grasp_detection_real_world(full_pcd)
            return grasp_poses, geometries, scores
            
        # Original implementation for simulation with known objects
        gg = self.compute_grasp_pose(full_pcd)
        del_index = []
        for index, value in enumerate(gg):
            if value.score < 0.15:
                del_index.append(index)
        for i in reversed(del_index):
            gg.remove(i)

        grasp_pose_set, grasp_pose_dict, remain_gg = self.assign_grasp_pose(gg, object_poses)
        
        return grasp_pose_set, grasp_pose_dict, remain_gg

