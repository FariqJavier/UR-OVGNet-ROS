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

        gg = self.graspnet_baseline.inference(grasp_pcd)
        if len(gg) == 0: # Early exit if no grasps from inference
            return gg

        gg.translations = -gg.translations
        gg.rotation_matrices = -gg.rotation_matrices # Still suspicious but keeping as per original

        if self.refine_approach_dist != 0.0:
             # Ensure gg.rotation_matrices is (N,3,3) for this broadcasting
             if gg.rotation_matrices.ndim == 3 and gg.rotation_matrices.shape[1:3] == (3,3):
                gg.translations = gg.translations + gg.rotation_matrices[:, :, 0] * self.refine_approach_dist
             else:
                 print(f"Warning: Unexpected shape for gg.rotation_matrices: {gg.rotation_matrices.shape}. Skipping refine_approach_dist.")

        gg = self.graspnet_baseline.collision_detection(gg, points)
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

    def filter_grasps_pose(self, gg, min_score=0.25, top_down_only=True):
        """Filter grasps by score, angle, and optionally top-down approach"""
        
        # Filter by score (removing low-score grasps)
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
        scores = gg.scores
        
        # Initialize a mask for filtering
        combined_mask = np.ones(len(rs), dtype=bool)
        
        # Filter for top-down grasping (approach from above)
        if top_down_only:
            top_down_angle_thresh = self.angle_thresh  # Max allowed angle from vertical
            approach_vectors = rs[:, :, 0]  # Approach direction in graspnet frame
            # Calculate the angle between approach vector and the negative Z-axis (in degrees)
            approach_angles = np.arccos(-approach_vectors[:, 2]) * 180.0 / np.pi
            # Mask for top-down approach
            approach_from_above_mask = approach_angles < top_down_angle_thresh
            combined_mask &= approach_from_above_mask
            rospy.loginfo(f"Filtered for top-down approach: kept {np.sum(approach_from_above_mask)} grasps.")

            if np.sum(combined_mask) < 1:
                rospy.logwarn("No top-down grasps found. Returning best available grasps.")
                sorted_indices = np.argsort(approach_angles)
                num_grasps = min(10, len(gg))  # Take the best 10 grasps
                filtered_gg = gg[sorted_indices[:num_grasps]]
                return filtered_gg, rs[sorted_indices[:num_grasps]]
                # return [], []

        # Final filtered grasps based on the combined mask
        filtered_gg = gg[combined_mask]
        
        # Calculate the angles for logging purposes
        approach_angles = np.arccos(-rs[:, 2, 0]) * 180.0 / np.pi
        
        # Log the range of approach angles for the selected grasps
        if np.sum(combined_mask) > 0:  # Only try to get min/max if we have values
            rospy.loginfo(f"Approach angle range of kept grasps: {np.min(approach_angles[combined_mask]):.1f}° to {np.max(approach_angles[combined_mask]):.1f}°")
        
        return filtered_gg, rs[combined_mask]

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

    def reorient_grasp_poses_vertical(self, grasp_poses, scores, geometries, num_best=10):
        """
        Reorient the best N grasp poses so that the approach direction is aligned 
        with the downward Z-axis and perpendicular to X and Y axes.
        
        Args:
            grasp_poses: List of grasp poses [x, y, z, qx, qy, qz, qw]
            scores: List of corresponding scores
            geometries: List of corresponding Open3D geometries
            num_best: Number of best poses to reorient (default: 10)
        
        Returns:
            reoriented_poses: List of reoriented grasp poses
            reoriented_geometries: List of reoriented geometries
            selected_scores: Scores of the selected poses
        """
        if len(grasp_poses) == 0:
            return [], [], []
        
        # Sort by scores (highest first) and select the best N
        sorted_indices = np.argsort(scores)[::-1]
        num_to_select = min(num_best, len(grasp_poses))
        best_indices = sorted_indices[:num_to_select]
        
        rospy.loginfo(f"Reorienting {num_to_select} best grasp poses for vertical approach")
        
        reoriented_poses = []
        reoriented_geometries = []
        selected_scores = []
        
        for idx in best_indices:
            original_pose = grasp_poses[idx]
            original_geometry = geometries[idx] if geometries else None
            score = scores[idx]
            
            # Extract position and original orientation
            position = original_pose[:3]
            original_quat = original_pose[3:]
            
            # Create new rotation matrix with downward Z approach
            # Approach direction: negative Z-axis (0, 0, 1)
            approach_dir = np.array([0, 0, 1])
            
            # For the gripper coordinate system, we need to define:
            # X-axis: approach direction (downward)
            # Y-axis: perpendicular to approach (we can choose based on original orientation)
            # Z-axis: perpendicular to both X and Y (right-hand rule)
            
            # Get the original rotation matrix to preserve some directional preference
            original_rotation = R.from_quat(original_quat).as_matrix()
            
            # Method 1: Keep the original Y direction projected onto the horizontal plane
            original_y = original_rotation[:, 1]  # Original Y-axis
            # Project original Y onto horizontal plane (remove Z component)
            horizontal_y = np.array([original_y[0], original_y[1], 0])
            
            # If the projection is too small, use a default direction
            if np.linalg.norm(horizontal_y) < 0.1:
                horizontal_y = np.array([1, 0, 0])  # Default to X direction
            
            # Normalize the horizontal Y direction
            y_axis = horizontal_y / np.linalg.norm(horizontal_y)
            
            # Calculate Z-axis using cross product (right-hand rule)
            # For downward approach: Z = X × Y
            x_axis = approach_dir  # (0, 0, 1)
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Recalculate Y to ensure orthogonality: Y = Z × X
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)

            # Ensure orthogonality
            assert np.isclose(np.linalg.norm(x_axis), 1), "X-axis is not normalized"
            assert np.isclose(np.linalg.norm(y_axis), 1), "Y-axis is not normalized"
            assert np.isclose(np.linalg.norm(z_axis), 1), "Z-axis is not normalized"

            assert np.isclose(np.dot(x_axis, y_axis), 0), "X and Y are not orthogonal"
            assert np.isclose(np.dot(y_axis, z_axis), 0), "Y and Z are not orthogonal"
            assert np.isclose(np.dot(z_axis, x_axis), 0), "Z and X are not orthogonal"
            
            # Create the new rotation matrix
            new_rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            # Ensure it's a proper rotation matrix (determinant = 1)
            if np.linalg.det(new_rotation_matrix) < 0:
                # Flip one axis to ensure right-handed coordinate system
                new_rotation_matrix[:, 1] = -new_rotation_matrix[:, 1]
            
            # Convert to quaternion
            new_rotation = R.from_matrix(new_rotation_matrix)
            new_quat = new_rotation.as_quat()
            
            # Create the reoriented pose
            reoriented_pose = np.concatenate([position, new_quat])
            reoriented_poses.append(reoriented_pose)
            selected_scores.append(score)
            
            # Transform the geometry if provided
            if original_geometry is not None:
                reoriented_geometry = copy.deepcopy(original_geometry)
                
                # Create transformation matrix for the geometry
                # First, inverse transform to origin using original orientation
                original_transform = np.eye(4)
                original_transform[:3, :3] = original_rotation
                original_transform[:3, 3] = position
                
                # New transform with reoriented rotation
                new_transform = np.eye(4)
                new_transform[:3, :3] = new_rotation_matrix
                new_transform[:3, 3] = position
                
                # Apply the transformation: T_new * T_original^(-1)
                combined_transform = new_transform @ np.linalg.inv(original_transform)
                reoriented_geometry.transform(combined_transform)
                
                reoriented_geometries.append(reoriented_geometry)
        
        rospy.loginfo(f"Successfully reoriented {len(reoriented_poses)} grasp poses with vertical approach")
        
        return reoriented_poses, reoriented_geometries, selected_scores

    def get_grasp_pose_center(self, grasp_pose):
        """Extract the grasp pose center from the grasp pose (position part)"""
        return grasp_pose[:3]  # [x, y, z]

    def get_surface_normal_and_points(self, object_pcd, grasp_pose_center, radius=0.05):
        """
        Calculate the surface normal and points of the object around the grasp pose center.
        
        Args:
            object_pcd: Open3D point cloud object
            grasp_pose_center: Grasp pose center (position)
            radius: Distance to search for surface points around the grasp pose center
            
        Returns:
            surface_normal: Normal vector of the surface
            surface_points: List of surface points within the radius
        """
        # Convert point cloud to numpy array
        points = np.asarray(object_pcd.points)
        
        # Filter the points to get surface points around the grasp pose center
        distances = np.linalg.norm(points - grasp_pose_center, axis=1)
        surface_points = points[distances < radius]
        
        # Calculate surface normal using PCA (principal component analysis)
        points_centered = surface_points - np.mean(surface_points, axis=0)
        cov_matrix = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # The eigenvector corresponding to the smallest eigenvalue is the surface normal
        surface_normal = eigenvectors[:, 0]  # This is the normal vector of the object surface
        
        return surface_normal, surface_points

    def adjust_grasp_pose_distance_to_surface(self, grasp_pose_center, surface_points, surface_normal, geometry, target_distance=0.02):
        """
        Adjust the grasp pose center so that it is exactly at a specified distance from the surface.

        Args:
            grasp_pose_center: The center of the grasp pose [x, y, z].
            surface_points: The points on the surface [Nx3 array of points].
            surface_normal: The normal vector of the surface [nx, ny, nz].
            target_distance: The target distance (default: 2 cm or 0.02 meters).

        Returns:
            adjusted_grasp_pose: The adjusted grasp pose with position and quaternion [x, y, z, qx, qy, qz, qw].
        """
        # Convert the surface points and surface normal to numpy arrays
        grasp_pose_center = np.array(grasp_pose_center)
        surface_points = np.array(surface_points)
        surface_normal = np.array(surface_normal)
        
        # Calculate distances from the grasp pose center to each surface point
        distances = np.linalg.norm(surface_points - grasp_pose_center, axis=1)
        
        # Find the closest surface point to the grasp pose center
        nearest_surface_point = surface_points[np.argmin(distances)]
        
        # Calculate the distance from the grasp pose center to the surface
        current_distance = np.min(distances)
        
        # Calculate the difference between the current distance and the target distance
        adjustment_distance = target_distance - current_distance
        
        # Adjust the grasp pose center by moving it along the surface normal
        adjusted_grasp_pose_center = grasp_pose_center + adjustment_distance * surface_normal
        
        # Create the reoriented grasp pose (assuming the orientation is unchanged)
        # Convert to quaternion (keeping the original orientation)
        # For simplicity, we assume that the original grasp pose orientation remains valid
        # You can adjust this if you want to change the orientation as well
        grasp_pose_quat = [0, 0, 0, 1]  # Placeholder quaternion (no rotation)
        
        # Combine position and quaternion
        adjusted_grasp_pose = np.concatenate([adjusted_grasp_pose_center, grasp_pose_quat])
        
        # Compute translation offset
        offset = adjusted_grasp_pose[:3] - grasp_pose_center

        # Adjust geometry accordingly
        geometry.translate(offset)

        return adjusted_grasp_pose, geometry
    
    def move_best_grasp_pose_to_object_midpoint(self, best_grasp_pose, object_pcd, best_grasp_geometry):
        """
        Move the best grasp pose to align its position with the object centroid.

        Args:
            best_grasp_pose: The best grasp pose [x, y, z, qx, qy, qz, qw]
            object_pcd: Open3D point cloud of the object
            best_grasp_geometry: The geometry of the best grasp pose
        
        Returns:
            moved_grasp_pose: Updated grasp pose with position at object centroid
        """
        # Extract point cloud data
        object_points = np.asarray(object_pcd.points)
            
        # Compute the mean for x, y coordinates
        mean_x = np.mean(object_points[:, 0])
        mean_y = np.mean(object_points[:, 1])
        object_centroid = [mean_x, mean_y]

        new_pose = np.array(best_grasp_pose)  # Copy
        new_pose[:2] = object_centroid

        offset = new_pose[:3] - best_grasp_pose[:3]

        best_grasp_geometry.translate(offset)

        return new_pose, best_grasp_geometry

    def move_best_grasp_pose_to_centroid(self, best_grasp_pose, object_pcd, best_grasp_geometry):
        """
        Move the best grasp pose to align its position with the object centroid.

        Args:
            best_grasp_pose: The best grasp pose [x, y, z, qx, qy, qz, qw]
            object_pcd: Open3D point cloud of the object
            best_grasp_geometry: The geometry of the best grasp pose
        
        Returns:
            moved_grasp_pose: Updated grasp pose with position at object centroid
        """
        # Extract point cloud data
        object_points = np.asarray(object_pcd.points)
        
        # Compute the centroid of the object point cloud
        object_centroid = np.mean(object_points, axis=0)

        new_pose = np.array(best_grasp_pose)
        new_pose[:3] = object_centroid  # Move the grasp pose to the centroid

        offset = new_pose[:3] - best_grasp_pose[:3]
        best_grasp_geometry.translate(offset)
        # Return the updated grasp pose and geometry
        return new_pose, best_grasp_geometry

    def select_best_grasp_near_centroid(self, grasp_poses, grasp_geometries, grasp_scores, object_pcd):
        """
        Selects the best grasp by finding the one closest to the object's centroid.

        This method combines two heuristics:
        1. High grasp quality score (already filtered).
        2. Stability (proximity to the center of mass).

        Args:
            grasp_poses: A list of candidate grasp poses [x, y, z, qx, qy, qz, qw].
            grasp_geometries: A list of corresponding Open3D geometries.
            grasp_scores: A list of corresponding scores.
            object_pcd: The Open3D point cloud of the target object.

        Returns:
            best_pose: The selected single best grasp pose.
            best_geometry: The corresponding geometry.
            best_score: The corresponding score.
        """
        if not grasp_poses:
            rospy.logwarn("No grasp poses provided to select from.")
            return None, None, None

        # 1. Calculate the object's 3D centroid
        object_points = np.asarray(object_pcd.points)
        if object_points.shape[0] == 0:
            rospy.logwarn("Cannot compute centroid of an empty point cloud.")
            # Fallback: return the highest-scoring grasp
            best_idx = np.argmax(grasp_scores)
            return grasp_poses[best_idx], grasp_geometries[best_idx], grasp_scores[best_idx]
            
        object_centroid = np.mean(object_points, axis=0)
        rospy.loginfo(f"Object centroid calculated at: {object_centroid}")
        
        # Visualize centroid for debugging
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        centroid_sphere.paint_uniform_color([1, 0, 0]) # Red
        centroid_sphere.translate(object_centroid)
        # o3d.visualization.draw_geometries([object_pcd, centroid_sphere] + grasp_geometries)


        # 2. Find the grasp pose closest to the centroid
        grasp_positions = np.array([pose[:3] for pose in grasp_poses])
        distances_to_centroid = np.linalg.norm(grasp_positions - object_centroid, axis=1)
        
        # 3. Select the index of the closest grasp
        closest_grasp_index = np.argmin(distances_to_centroid)
        rospy.loginfo(f"Selected grasp #{closest_grasp_index} as it is closest to the centroid.")
        
        best_pose = grasp_poses[closest_grasp_index]
        best_geometry = grasp_geometries[closest_grasp_index]
        best_score = grasp_scores[closest_grasp_index]
        
        return best_pose, best_geometry, best_score

    def select_best_grasp_near_midpoint(self, grasp_poses, grasp_geometries, grasp_scores, object_pcd):
        """
        Selects the best grasp by finding the one closest to the geometric center 
        of the object's axis-aligned bounding box (midpoint).
        """
        if not grasp_poses:
            rospy.logwarn("No grasp poses provided to select from.")
            return None, None, None

        object_points = np.asarray(object_pcd.points)
        if object_points.shape[0] == 0:
            rospy.logwarn("Cannot compute midpoint of an empty point cloud.")
            best_idx = np.argmax(grasp_scores)
            return grasp_poses[best_idx], grasp_geometries[best_idx], grasp_scores[best_idx]
            
        # 1. CORRECT: Calculate the true 3D geometric midpoint
        min_coords = np.min(object_points, axis=0)
        max_coords = np.max(object_points, axis=0)
        object_midpoint_3d = (min_coords + max_coords) / 2.0
        rospy.loginfo(f"Object 3D midpoint (bounding box center) calculated at: {object_midpoint_3d}")

        # 2. Find the grasp pose closest to the 3D midpoint
        grasp_positions = np.array([pose[:3] for pose in grasp_poses])
        distances_to_midpoint = np.linalg.norm(grasp_positions - object_midpoint_3d, axis=1)
        
        # 3. Select the index of the closest grasp
        closest_grasp_index = np.argmin(distances_to_midpoint)
        rospy.loginfo(f"Selected grasp #{closest_grasp_index} as it is closest to the 3D midpoint.")
        
        best_pose = grasp_poses[closest_grasp_index]
        best_geometry = grasp_geometries[closest_grasp_index]
        best_score = grasp_scores[closest_grasp_index]
        
        return best_pose, best_geometry, best_score
        
    def grasp_detection_real_world(self, fused_pcd_world, fused_pcd_canonical, world_to_canonical_transform, get_visual, min_score=0.25, top_down_only=True, num_best=10, simple_orientation=True):
        """
        Generate grasping poses for a real-world setting without known object poses.
        
        Args:
            fused_pcd_world: Point cloud in world coordinates
            fused_pcd_canonical: Point cloud in canonical coordinates  
            world_to_canonical_transform: Transformation matrix
            get_visual: Whether to show visualizations
            min_score: Minimum score threshold
            num_best: Number of best grasps to reorient
            simple_orientation: Use simple standard orientation vs. adaptive orientation
        
        Returns:
            reoriented_poses: List of vertically oriented grasp poses
            reoriented_geometries: Corresponding visualization geometries
            scores: Grasp scores
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

            if len(grasp_poses_canonical) == 0:
                rospy.logwarn("No grasp poses found for reorientation")
                return [], [], []

            # Reorient the best poses vertically
            grasp_pose_reoriented, geometries_reoriented, scores_reoriented = self.reorient_grasp_poses_vertical(
                grasp_poses_canonical, scores_canonical, geometries_canonical, num_best
            )

            if len(grasp_pose_reoriented) == 0:
                rospy.logwarn("No grasp poses found for reorientation")
                return [], [], []

            for i in range(len(grasp_pose_reoriented)):
                # Get the center of the grasp pose
                grasp_pose_center = self.get_grasp_pose_center(grasp_pose_reoriented[i])
                # Get the surface normal and points around the grasp pose center
                object_surface_normal, object_surface_points = self.get_surface_normal_and_points(fused_pcd_canonical, grasp_pose_center, radius=0.05)
                # Reorient the grasp pose to be at a distance of 2 cm from the surface
                adjusted_grasp_pose, adjusted_geometry = self.adjust_grasp_pose_distance_to_surface(
                    grasp_pose_center, object_surface_points, object_surface_normal, geometries_reoriented[i], target_distance=0.02
                )
                # Calculate distances from the grasp pose center to each surface point
                distances = np.linalg.norm(object_surface_points - adjusted_grasp_pose[:3], axis=1)
                rospy.loginfo(f"Adjusted grasp pose {i} center to object surface, distances to surface points:")
                # Update the reoriented grasp pose with the adjusted position
                grasp_pose_reoriented[i][:3] = adjusted_grasp_pose[:3]
                # Update the geometry with the adjusted position
                geometries_reoriented[i] = adjusted_geometry

            # # Adjust the best grasp pose to the object midpoint
            # grasp_pose_reoriented, geometries_reoriented = self.move_best_grasp_pose_to_object_midpoint(
            #     grasp_pose_reoriented[0], fused_pcd_canonical, geometries_reoriented[0]
            # )

            # # Adjust the best grasp pose to the object centroid
            # grasp_pose_reoriented[0], geometries_reoriented[0] = self.move_best_grasp_pose_to_centroid(
            #     grasp_pose_reoriented[0], fused_pcd_canonical, geometries_reoriented[0]
            # )

            best_pose, best_geometry, best_score = self.select_best_grasp_near_centroid(
                grasp_pose_reoriented, geometries_reoriented, scores_reoriented, fused_pcd_canonical
            )

            # best_pose, best_geometry, best_score = self.select_best_grasp_near_midpoint(
            #     grasp_pose_reoriented, geometries_reoriented, scores_reoriented, fused_pcd_canonical
            # )

            if best_pose is None:
                rospy.logwarn("Could not select a best grasp pose.")
                return [], [], []

            # Optional visualization
            if get_visual:  # Set to True for debugging
                # Color the grasp poses by score
                geometries_canonical = self.color_grasp_poses_by_score(geometries_canonical, scores_canonical)
                geometries_reoriented = self.color_grasp_poses_by_score(geometries_reoriented, scores_reoriented)
                # Visualize the canonical based grasp poses
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([frame, fused_pcd_canonical] + geometries_canonical, f'Canonical Grasp Poses')
                # Visualize the world based grasp poses
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([frame, fused_pcd_canonical] + geometries_reoriented, f'Reoriented Grasp Poses')
                
            # return grasp_pose_reoriented, geometries_reoriented, scores_reoriented
            return [best_pose], [best_geometry], [best_score]
            
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

