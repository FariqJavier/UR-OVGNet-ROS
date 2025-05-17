import numpy as np
import open3d as o3d
import open3d_plus as o3dp
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler
import copy
import os
import sys
import rospkg
import rospy

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

    def filter_grasps_pose(self, gg, min_score=0.15, top_down_only=True):
        """Filter grasps by score, distance, angle, and optionally top-down approach"""
        
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

        # Initialize a mask for filtering
        combined_mask = np.ones(len(rs), dtype=bool)

        if self.refine_approach_dist is not None:
            # Apply refine_approach_dist filtering by checking the approach direction and depth
            depth_mask = (depths > self.refine_approach_dist)
            combined_mask &= depth_mask
            rospy.loginfo(f"Filtered by refine_approach_dist: kept {np.sum(depth_mask)} grasps.")

        if self.dist_thresh is not None:
            # Apply distance threshold filtering based on distance to the object center
            min_dists = np.linalg.norm(ts - np.mean(ts, axis=0), axis=1)
            dist_mask = min_dists < self.dist_thresh
            combined_mask &= dist_mask
            rospy.loginfo(f"Filtered by dist_thresh: kept {np.sum(dist_mask)} grasps.")

        if self.angle_thresh is not None:
            # Calculate angles between approach direction and vertical (in degrees)
            approach_angles = np.arccos(-rs[:, 2, 0]) * 180.0 / np.pi
            angle_mask = approach_angles < self.angle_thresh
            combined_mask &= angle_mask
            rospy.loginfo(f"Filtered by angle_thresh (<{self.angle_thresh}°): kept {np.sum(angle_mask)} grasps.")
            if np.sum(angle_mask) > 0:  # Only try to get min/max if we have values
                rospy.loginfo(f"Angle range of kept grasps: {np.min(approach_angles[angle_mask]):.1f}° to {np.max(approach_angles[angle_mask]):.1f}°")
            else:
                rospy.loginfo("No grasps within the angle threshold.")

        if top_down_only:
            # Stricter filtering for top-down grasps (approach from above)
            top_down_angle_thresh = min(self.angle_thresh, 45) # degrees (adjust as needed)
            approach_vectors = rs[:, :, 0]  # Approach direction in graspnet frame
            # Calculate the angle between approach vector and negative Z-axis (in degrees)
            # A perfect top-down grasp would have angle = 0
            # approach_from_above_mask = approach_vectors[:, 2] < -0.8  # z-component should be negative
            approach_angles = np.arccos(-approach_vectors[:, 2]) * 180.0 / np.pi
            approach_from_above_mask = approach_angles < top_down_angle_thresh
            combined_mask &= approach_from_above_mask
            rospy.loginfo(f"Filtered for top-down approach: kept {np.sum(approach_from_above_mask)} grasps.")

            if np.sum(combined_mask) < 1:
                rospy.logwarn("No top-down grasps found. Using best available grasps...")
                # Fall back to best angle grasps
                # angles = np.arccos(-rs[:, 2, 0]) * 180.0 / np.pi
                # sorted_indices = np.argsort(angles)
                sorted_indices = np.argsort(approach_angles)
                num_grasps = min(10, len(gg))
                filtered_gg = gg[sorted_indices[:num_grasps]]
                return filtered_gg, eelink_rs[sorted_indices[:num_grasps]]
            else:
                # # Use grasps that pass all the filters
                # filtered_gg = gg[combined_mask]
                # sorted_indices = np.argsort(-scores[combined_mask])
                # num_grasps = min(20, len(filtered_gg))
                # filtered_gg = filtered_gg[sorted_indices[:num_grasps]]
                # return filtered_gg, eelink_rs[combined_mask][sorted_indices[:num_grasps]]

                # Use grasps that pass all the filters
                filtered_gg = gg[combined_mask]
                
                # Calculate combined scores that prioritize top-down grasps
                top_down_scores = 1.0 - (approach_angles[combined_mask] / top_down_angle_thresh)
                
                # Weight between original grasp score and top-down score
                # Adjust these weights to balance grasp quality vs. top-down preference
                original_weight = 0.3
                top_down_weight = 0.7
                
                # Calculate combined scores (higher is better)
                combined_scores = (original_weight * scores[combined_mask]) + \
                                (top_down_weight * top_down_scores)
                
                # Sort by combined score
                sorted_indices = np.argsort(-combined_scores)
                num_grasps = min(20, len(filtered_gg))
                filtered_gg = filtered_gg[sorted_indices[:num_grasps]]
                
                # Log the selected grasps for debugging
                rospy.loginfo(f"Selected {len(filtered_gg)} grasps with top-down priority")
                if len(filtered_gg) > 0:
                    best_angle = approach_angles[combined_mask][sorted_indices[0]]
                    best_score = scores[combined_mask][sorted_indices[0]]
                    rospy.loginfo(f"Best grasp: angle = {best_angle:.2f}°, original score = {best_score:.2f}")
                
                return filtered_gg, eelink_rs[combined_mask][sorted_indices[:num_grasps]]
        else:
            # For taller objects, allow side grasps but prefer grasps on the object body, not edges
            # Filter by score first
            sorted_indices = np.argsort(-scores)
            num_grasps = min(30, len(gg))  # Keep more candidates for tall objects
            filtered_gg = gg[sorted_indices[:num_grasps]]
            
            # Allow wider range of approach angles but avoid extreme angles
            approach_vectors = rs[sorted_indices[:num_grasps], :, 0]
            approach_angles = np.arccos(np.abs(approach_vectors[:, 2])) * 180.0 / np.pi
            
            # Sort by a combination of grasp score and reasonable approach angle
            combined_scores = scores[sorted_indices[:num_grasps]] * (1.0 - approach_angles/90.0)
            final_indices = np.argsort(-combined_scores)
            
            final_gg = filtered_gg[final_indices]
            return final_gg, eelink_rs[sorted_indices[:num_grasps]][final_indices]

    def convert_grasps_to_poses(self, gg, eelink_rs):
        """Convert GraspGroup to a list of grasp poses (position + quaternion)"""
        grasp_poses = []
        geometries = []
        scores = []
        
        # Get the arrays from the GraspGroup
        translations = gg.translations
        rotation_matrices = gg.rotation_matrices
        depths = gg.depths
        gg_scores = gg.scores
        
        for i in range(len(gg)):
            # Add visualization geometry
            geometries.append(gg[i].to_open3d_geometry())
            scores.append(gg_scores[i])
            
            # Get grasp rotation matrix
            grasp_rotation_matrix = eelink_rs[i]
            
            # Ensure right-hand coordinate system
            if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                grasp_rotation_matrix[:,0] = -grasp_rotation_matrix[:, 0]
            
            # Convert to position + quaternion format
            grasp_pose = np.zeros(7)
            grasp_pose[:3] = translations[i] + rotation_matrices[i,:,0] * depths[i]  # Position
            r = R.from_matrix(grasp_rotation_matrix)
            grasp_pose[3:] = r.as_quat()  # Quaternion
            
            grasp_poses.append(grasp_pose)
        
        return grasp_poses, geometries, scores

    def color_grasp_poses_by_score(self, geometries, scores):
        """
        Color grasp poses by their score. The higher the score, the greener the color.

        Args:
            geometries: List of Open3D geometries representing the grasp poses
            scores: List of grasp scores corresponding to each grasp pose

        Returns:
            geometries: List of Open3D geometries with updated colors
        """
        # Normalize the scores to range [0, 1] using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()

        # Map normalized scores to a color map (e.g., green to red color map)
        for i, geometry in enumerate(geometries):
            score = normalized_scores[i]
            
            # Map score to a color using a simple linear gradient from red (low) to green (high)
            color = np.array([1.0 - score, score, 0.0])  # [R, G, B]
            
            # Apply color to the geometry
            geometry.paint_uniform_color(color)

        return geometries
    
    def grasp_detection_real_world(self, fused_pcd_world, fused_pcd_canonical, world_to_canonical_transform, get_visual, min_score=0.15, top_down_only=True):
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
            if fused_pcd_canonical is None or len(fused_pcd_canonical.points) < 10:
                rospy.logwarn("Invalid or empty point cloud for grasp detection")
                return [], [], []
            
            # Compute grasp candidates
            rospy.loginfo(f"Computing grasp poses on point cloud with {len(fused_pcd_canonical.points)} points")
            gg = self.compute_grasp_pose(fused_pcd_canonical)
            
            # Log number of grasps found
            rospy.loginfo(f"Found {len(gg)} grasp candidates before filtering")
            
            if len(gg) == 0:
                rospy.logwarn("No grasp candidates found")
                return [], [], []
            
            # Filter grasps by score and angle
            # filtered_gg, eelink_rs = self.filter_grasps_by_score_and_angle(gg, min_score, top_down_only)
            filtered_gg, eelink_rs = self.filter_grasps_pose(gg, min_score, top_down_only)
            
            # Log number of grasps after filtering
            rospy.loginfo(f"Filtered to {len(filtered_gg)} grasp candidates")
            
            if len(filtered_gg) == 0:
                rospy.logwarn("No grasp candidates remain after filtering")
                return [], [], []
            
            # Convert to grasp poses
            grasp_poses_canonical, geometries_canonical, scores_canonical = self.convert_grasps_to_poses(filtered_gg, eelink_rs)

            # Transform grasps back to world coordinates
            canonical_to_world_transform = np.linalg.inv(world_to_canonical_transform)
            grasp_poses_world = []
            geometries_world = []
            scores_world = []

            for grasp_pose_canonical, geometry_canonical, score_canonical in zip(grasp_poses_canonical, geometries_canonical, scores_canonical):
                # Split the canonical grasp pose into position and quaternion (rotation)
                position_canonical = grasp_pose_canonical[:3]
                quaternion_canonical = grasp_pose_canonical[3:]

                # Apply the transformation to the position
                position_world = canonical_to_world_transform[:3, :3] @ position_canonical + canonical_to_world_transform[:3, 3]

                # Apply the transformation to the quaternion (rotation)
                rotation_matrix = R.from_quat(quaternion_canonical).as_matrix()
                rotation_world = canonical_to_world_transform[:3, :3] @ rotation_matrix

                # Convert the rotation matrix back to a quaternion
                quaternion_world = R.from_matrix(rotation_world).as_quat()

                # Create the final grasp pose in world coordinates
                grasp_pose_world = np.concatenate([position_world, quaternion_world])

                geometry_world = copy.deepcopy(geometry_canonical)
                geometry_world.transform(canonical_to_world_transform)

                grasp_poses_world.append(grasp_pose_world)
                geometries_world.append(geometry_world)
                scores_world.append(score_canonical)
            
            # Optional visualization
            if get_visual:  # Set to True for debugging
                # Color the grasp poses by score
                geometries_canonical = self.color_grasp_poses_by_score(geometries_canonical, scores_canonical)
                geometries_world = self.color_grasp_poses_by_score(geometries_world, scores_world)
                # Visualize the canonical based grasp poses
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([frame, fused_pcd_canonical] + geometries_canonical, f'Canonical Grasp Poses')
                # Visualize the world based grasp poses
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([frame, fused_pcd_world] + geometries_world, f'World Grasp Poses')
                
            return grasp_poses_world, geometries_world, scores_world
            
        except Exception as e:
            rospy.logerr(f"Error in grasp detection: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return [], [], []

    def grasp_detection_real_world_multiview(self, fused_pcd_world, fused_pcd_canonical, world_to_canonical_transform, get_visual, camera_id, min_score=0.15, top_down_only=True):
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
            if fused_pcd_canonical is None or len(fused_pcd_canonical.points) < 10:
                rospy.logwarn("Invalid or empty point cloud for grasp detection")
                return [], [], []
            
            # Compute grasp candidates
            rospy.loginfo(f"Computing grasp poses on point cloud with {len(fused_pcd_canonical.points)} points")
            gg = self.compute_grasp_pose(fused_pcd_canonical)
            
            # Log number of grasps found
            rospy.loginfo(f"Found {len(gg)} grasp candidates before filtering")
            
            if len(gg) == 0:
                rospy.logwarn("No grasp candidates found")
                return [], [], []
            
            # Filter grasps by score and angle
            # filtered_gg, eelink_rs = self.filter_grasps_by_score_and_angle(gg, min_score, top_down_only)
            filtered_gg, eelink_rs = self.filter_grasps_pose(gg, min_score, top_down_only)
            
            # Log number of grasps after filtering
            rospy.loginfo(f"Filtered to {len(filtered_gg)} grasp candidates")
            
            if len(filtered_gg) == 0:
                rospy.logwarn("No grasp candidates remain after filtering")
                return [], [], []
            
            # Convert to grasp poses
            grasp_poses_canonical, geometries_canonical, scores_canonical = self.convert_grasps_to_poses(filtered_gg, eelink_rs)

            # Transform grasps back to world coordinates
            canonical_to_world_transform = np.linalg.inv(world_to_canonical_transform)
            grasp_poses_world = []
            geometries_world = []
            scores_world = []

            for grasp_pose_canonical, geometry_canonical, score_canonical in zip(grasp_poses_canonical, geometries_canonical, scores_canonical):
                # Split the canonical grasp pose into position and quaternion (rotation)
                position_canonical = grasp_pose_canonical[:3]
                quaternion_canonical = grasp_pose_canonical[3:]

                # Apply the transformation to the position
                position_world = canonical_to_world_transform[:3, :3] @ position_canonical + canonical_to_world_transform[:3, 3]

                # Apply the transformation to the quaternion (rotation)
                rotation_matrix = R.from_quat(quaternion_canonical).as_matrix()
                rotation_world = canonical_to_world_transform[:3, :3] @ rotation_matrix

                # Convert the rotation matrix back to a quaternion
                quaternion_world = R.from_matrix(rotation_world).as_quat()

                # Create the final grasp pose in world coordinates
                grasp_pose_world = np.concatenate([position_world, quaternion_world])

                geometry_world = copy.deepcopy(geometry_canonical)
                geometry_world.transform(canonical_to_world_transform)

                grasp_poses_world.append(grasp_pose_world)
                geometries_world.append(geometry_world)
                scores_world.append(score_canonical)
            
            # Optional visualization
            if get_visual:  # Set to True for debugging
                # Color the grasp poses by score
                geometries_canonical = self.color_grasp_poses_by_score(geometries_canonical, scores_canonical)
                geometries_world = self.color_grasp_poses_by_score(geometries_world, scores_world)
                # Visualize the canonical based grasp poses
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([frame, fused_pcd_canonical] + geometries_canonical, f'Camera {camera_id} - Canonical Grasp Poses')
                # Visualize the world based grasp poses
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([frame, fused_pcd_world] + geometries_world, f'Camera {camera_id} - World Grasp Poses')
                
            return grasp_poses_world, geometries_world, scores_world
            
        except Exception as e:
            rospy.logerr(f"Error in grasp detection: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return [], [], []

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

