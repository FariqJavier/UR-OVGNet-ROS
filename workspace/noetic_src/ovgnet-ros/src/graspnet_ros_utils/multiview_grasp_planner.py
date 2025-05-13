import copy
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
from collections import namedtuple
import open3d as o3d
import pybullet as p
from graspnet_ros_utils.processing_graspnet import (
    get_single_pointcloud,
    get_single_pointcloud_fixed,
    debug_point_cloud_pipeline
)

# Define a structure for grasp candidates
GraspCandidate = namedtuple('GraspCandidate', [
    'pose', 'score', 'camera_id', 'confidence', 'distance_to_robot', 
    'view_angle', 'reachability_score', 'geometry', 'approach_angle'
])

class MultiViewGraspPlanner:
    """Multi-view grasp planner optimized for eye-in-hand configuration"""
    
    def __init__(self, graspnet_instance, robot_tf_listener=None, robot_base_frame='world'):
        """
        Initialize eye-in-hand grasp planner
        
        Args:
            graspnet_instance: Initialized Graspnet object
            robot_tf_listener: TF listener for coordinate transforms
            robot_base_frame: Robot base frame name (default: 'world')
        """
        self.graspnet = graspnet_instance
        self.robot_tf_listener = robot_tf_listener
        self.robot_base_frame = robot_base_frame
        
        # Eye-in-hand specific parameters
        self.min_points_required = 100
        self.max_grasps_per_view = 20
        self.min_grasp_score = 0.15
        self.top_down_only = True
        
        # Eye-in-hand specific constraints
        self.max_approach_angle = 45.0  # Maximum angle from vertical (degrees)
        self.enforce_surface_alignment = True
        self.surface_offset = 0.002  # 2mm offset from surface
        self.min_height_above_table = 0.005  # Minimum height above table
        
    def plan_multiview_grasps(self, realsense_inputs, detection_results, get_visual=False):
        """
        Generate grasp candidates optimized for eye-in-hand configuration
        
        Args:
            realsense_inputs: Dict of camera_id -> realsense input data
            detection_results: Dict of camera_id -> detection results
            get_visual: Whether to show visualization
            
        Returns:
            List of GraspCandidate objects sorted by quality score
        """
        all_grasp_candidates = []
        
        rospy.loginfo(f"Starting eye-in-hand grasp planning for {len(realsense_inputs)} cameras")
        
        # Analyze camera setup for eye-in-hand optimization
        camera_analysis = self._analyze_camera_setup(realsense_inputs)
        
        # Process each camera view
        for camera_id, realsense_input in realsense_inputs.items():
            if camera_id not in detection_results:
                rospy.logwarn(f"No detection result for camera {camera_id}")
                continue
                
            rospy.loginfo(f"Processing grasps for camera {camera_id}")
            
            # Generate point cloud for this view
            # debug_point_cloud_pipeline(realsense_inputs, detection_results)
            # pcd = get_single_pointcloud(realsense_input, detection_results[camera_id], camera_id)
            pcd = get_single_pointcloud_fixed(realsense_input, detection_results[camera_id], camera_id)
            
            if pcd is None or len(pcd.points) < self.min_points_required:
                rospy.logwarn(f"Camera {camera_id}: Insufficient point cloud quality")
                continue
            
            # Apply eye-in-hand specific preprocessing
            pcd = self._preprocess_eyeinhand_pointcloud(pcd, realsense_input.camera_info)
            
            # Calculate eye-in-hand specific view metrics
            view_metrics = self._calculate_eyeinhand_view_metrics(
                pcd, realsense_input.camera_info, camera_id, camera_analysis
            )
            
            # Generate grasps for this view
            try:
                grasp_poses, geometries, scores = self.graspnet.grasp_detection_real_world(
                    pcd, 
                    get_visual=get_visual,
                    min_score=self.min_grasp_score,
                    top_down_only=self.top_down_only
                )
                
                rospy.loginfo(f"Camera {camera_id}: Found {len(grasp_poses)} raw grasps")
                
                # Apply eye-in-hand specific grasp processing
                processed_grasps = self._process_eyeinhand_grasps(
                    grasp_poses, geometries, scores, pcd, realsense_input.camera_info
                )
                
                rospy.loginfo(f"Camera {camera_id}: {len(processed_grasps)} grasps after processing")
                
                # Convert to grasp candidates
                for grasp_data in processed_grasps:
                    candidate = self._create_eyeinhand_grasp_candidate(
                        grasp_data, camera_id, view_metrics, realsense_input.camera_info
                    )
                    
                    if candidate and self._validate_eyeinhand_grasp(candidate):
                        all_grasp_candidates.append(candidate)
                        
            except Exception as e:
                rospy.logerr(f"Error generating grasps for camera {camera_id}: {str(e)}")
                import traceback
                rospy.logerr(traceback.format_exc())
                continue
        
        # Rank grasps with eye-in-hand specific criteria
        ranked_grasps = self._rank_eyeinhand_grasps(all_grasp_candidates)
        
        rospy.loginfo(f"Generated {len(ranked_grasps)} valid grasp candidates")
        
        # Visualization specific to eye-in-hand setup
        if get_visual and ranked_grasps:
            self._visualize_eyeinhand_grasps(ranked_grasps, realsense_inputs)
        
        return ranked_grasps
    
    def _analyze_camera_setup(self, realsense_inputs):
        """Analyze camera setup to understand spatial relationships"""
        camera_analysis = {}
        
        # Calculate average camera height and spread
        heights = []
        positions = []
        
        for camera_id, input_data in realsense_inputs.items():
            pos = np.array(input_data.camera_info["position"])
            heights.append(pos[2])
            positions.append(pos)
        
        camera_analysis['average_height'] = np.mean(heights)
        camera_analysis['height_std'] = np.std(heights)
        camera_analysis['positions'] = positions
        
        # Determine if all cameras are above the workspace
        camera_analysis['all_above_table'] = np.min(heights) > 0.05  # 5cm above origin
        
        rospy.loginfo(f"Camera setup analysis: avg_height={camera_analysis['average_height']:.3f}m, "
                     f"all_above_table={camera_analysis['all_above_table']}")
        
        return camera_analysis
    
    def _preprocess_eyeinhand_pointcloud(self, pcd, camera_info):
        """Apply eye-in-hand specific preprocessing to point cloud"""
        if len(pcd.points) < 50:
            return pcd
        
        # More aggressive outlier removal for eye-in-hand (closer cameras = more noise)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        
        # Ensure object is on table plane for eye-in-hand cameras
        points = np.asarray(pcd.points)
        min_z = np.min(points[:, 2])
        
        # If object appears floating (common with eye-in-hand due to viewing angle)
        if min_z > self.min_height_above_table:
            rospy.loginfo(f"Adjusting object to table height (was {min_z:.3f}m above)")
            pcd.translate([0, 0, -(min_z - self.min_height_above_table)])
        
        # Apply additional smoothing for eye-in-hand data
        if hasattr(pcd, 'voxel_down_sample'):
            pcd = pcd.voxel_down_sample(voxel_size=0.002)  # 2mm voxels
        
        # Estimate normals with parameters optimized for eye-in-hand
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.01, max_nn=50
            )
        )
        
        return pcd
    
    def _calculate_eyeinhand_view_metrics(self, pcd, camera_info, camera_id, camera_analysis):
        """Calculate view quality metrics specific to eye-in-hand setup"""
        points = np.asarray(pcd.points)
        camera_pos = np.array(camera_info["position"])
        
        metrics = {
            'point_count': len(points),
            'camera_position': camera_pos,
            'camera_id': camera_id
        }
        
        # Calculate object bounding box
        bbox = pcd.get_axis_aligned_bounding_box()
        object_center = bbox.get_center()
        object_extent = bbox.get_extent()
        
        # Eye-in-hand specific metrics
        metrics['object_center'] = object_center
        metrics['object_extent'] = object_extent
        
        # Height advantage (how much camera is above object)
        metrics['height_advantage'] = camera_pos[2] - object_center[2]
        
        # Viewing angle from vertical
        view_vector = object_center - camera_pos
        vertical = np.array([0, 0, -1])  # Downward is good for eye-in-hand
        
        view_angle = np.arccos(np.clip(
            np.dot(view_vector, vertical) / (np.linalg.norm(view_vector) * np.linalg.norm(vertical)),
            -1, 1
        )) * 180 / np.pi
        
        metrics['view_angle'] = view_angle
        
        # Quality score for this view (combine multiple factors)
        metrics['view_quality'] = self._calculate_view_quality_score(metrics)
        
        return metrics
    
    def _calculate_view_quality_score(self, metrics):
        """Calculate overall view quality score for eye-in-hand setup"""
        score = 1.0
        
        # Prefer cameras that are above the object
        if metrics['height_advantage'] > 0.1:  # 10cm above
            score *= 1.2
        elif metrics['height_advantage'] < 0:  # Below object (bad)
            score *= 0.3
        
        # Prefer near-vertical viewing angles (good for top-down grasps)
        if metrics['view_angle'] < 30:  # Less than 30 degrees from vertical
            score *= 1.1
        elif metrics['view_angle'] > 60:  # More than 60 degrees from vertical
            score *= 0.7
        
        # Point cloud density bonus
        if metrics['point_count'] > 500:
            score *= 1.05
        
        return score
    
    def _process_eyeinhand_grasps(self, grasp_poses, geometries, scores, pcd, camera_info):
        """Process grasps with eye-in-hand specific modifications"""
        processed_grasps = []
        points = np.asarray(pcd.points)
        
        for i, (grasp_pose, geometry, score) in enumerate(zip(grasp_poses, geometries, scores)):
            # Convert to matrix format
            grasp_matrix = self._pose_to_matrix(grasp_pose)
            
            # Apply eye-in-hand specific corrections
            if self.enforce_surface_alignment:
                grasp_matrix = self._align_grasp_to_surface(grasp_matrix, points)
            
            grasp_matrix = self._enforce_downward_approach(grasp_matrix)
            
            # Validate approach angle
            approach_angle = self._calculate_approach_angle(grasp_matrix)
            if approach_angle > self.max_approach_angle:
                rospy.logdebug(f"Skipping grasp {i} with approach angle {approach_angle:.1f}°")
                continue
            
            processed_grasps.append({
                'pose_matrix': grasp_matrix,
                'original_pose': grasp_pose,
                'score': score,
                'geometry': geometry,
                'approach_angle': approach_angle
            })
        
        return processed_grasps
    
    def _align_grasp_to_surface(self, grasp_matrix, points):
        """Ensure grasp is properly aligned with object surface"""
        grasp_pos = grasp_matrix[:3, 3]
        
        # Find closest point on surface
        distances = np.linalg.norm(points - grasp_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = points[closest_idx]
        min_distance = distances[closest_idx]
        
        # If grasp is too far from surface, move it
        if min_distance > self.surface_offset:
            # Get surface normal at closest point (approximate)
            # Use multiple nearby points to estimate normal
            k = min(10, len(points))
            closest_indices = np.argpartition(distances, k)[:k]
            nearby_points = points[closest_indices]
            
            # Estimate surface normal using PCA
            centered_points = nearby_points - np.mean(nearby_points, axis=0)
            _, _, vh = np.linalg.svd(centered_points)
            surface_normal = vh[2]  # Smallest component
            
            # Ensure normal points outward (away from camera/grasp)
            if np.dot(surface_normal, grasp_pos - closest_point) < 0:
                surface_normal = -surface_normal
            
            # Place grasp at surface with small offset
            new_pos = closest_point + surface_normal * self.surface_offset
            grasp_matrix[:3, 3] = new_pos
        
        return grasp_matrix
    
    def _enforce_downward_approach(self, grasp_matrix):
        """Ensure grasp approaches from above (good for eye-in-hand)"""
        approach_dir = grasp_matrix[:3, 2]  # Z-axis is approach direction
        
        # If approach is upward, flip it
        if approach_dir[2] > 0:
            # Flip the Z-axis
            grasp_matrix[:3, 2] = -approach_dir
            # Adjust Y-axis to maintain right-handed coordinate system
            grasp_matrix[:3, 1] = -grasp_matrix[:3, 1]
        
        # Ensure approach is reasonably vertical
        vertical_component = -approach_dir[2]  # We want negative Z
        if vertical_component < 0.7:  # Less than ~45 degrees from horizontal
            # Force approach to be more vertical
            new_z = np.array([0, 0, -1])
            
            # Keep X in XY plane
            new_x = grasp_matrix[:3, 0].copy()
            new_x[2] = 0
            new_x = new_x / np.linalg.norm(new_x)
            
            # Calculate Y as cross product
            new_y = np.cross(new_z, new_x)
            
            # Update rotation matrix
            grasp_matrix[:3, :3] = np.column_stack([new_x, new_y, new_z])
        
        return grasp_matrix
    
    def _calculate_approach_angle(self, grasp_matrix):
        """Calculate angle between grasp approach and vertical"""
        approach_dir = grasp_matrix[:3, 2]
        vertical = np.array([0, 0, -1])  # Downward
        
        angle = np.arccos(np.clip(np.dot(approach_dir, vertical), -1, 1)) * 180 / np.pi
        return angle
    
    def _create_eyeinhand_grasp_candidate(self, grasp_data, camera_id, view_metrics, camera_info):
        """Create a grasp candidate with eye-in-hand specific processing"""
        try:
            # Transform to robot base frame
            robot_pose = self._transform_grasp_to_robot_frame(grasp_data['pose_matrix'], camera_info)
            
            if robot_pose is None:
                return None
            
            # Calculate eye-in-hand specific metrics
            distance_to_robot = np.linalg.norm(robot_pose[:3, 3])
            reachability_score = self._calculate_eyeinhand_reachability(robot_pose)
            
            # Enhanced scoring for eye-in-hand
            combined_score = self._calculate_eyeinhand_combined_score(
                grasp_data['score'], view_metrics, distance_to_robot, 
                reachability_score, grasp_data['approach_angle']
            )
            
            return GraspCandidate(
                pose=robot_pose,
                score=combined_score,
                camera_id=camera_id,
                confidence=grasp_data['score'],
                distance_to_robot=distance_to_robot,
                view_angle=view_metrics['view_angle'],
                reachability_score=reachability_score,
                geometry=grasp_data['geometry'],
                approach_angle=grasp_data['approach_angle']
            )
            
        except Exception as e:
            rospy.logerr(f"Error creating grasp candidate: {str(e)}")
            return None
    
    def _calculate_eyeinhand_reachability(self, grasp_pose):
        """Calculate reachability score considering eye-in-hand constraints"""
        position = grasp_pose[:3, 3]
        
        # Eye-in-hand specific workspace limits
        xy_distance = np.linalg.norm(position[:2])
        height = position[2]
        
        # Adjusted limits for eye-in-hand (typically closer to robot)
        optimal_distance = 0.4  # Closer optimal distance
        max_distance = 0.8      # Closer max distance
        min_distance = 0.15     # Closer min distance
        max_height = 0.8        # Lower max height
        min_height = -0.05      # Allow slightly below table
        
        # Distance score (prefer closer grasps for eye-in-hand)
        if xy_distance < min_distance or xy_distance > max_distance:
            distance_score = 0
        else:
            distance_score = 1.0 - abs(xy_distance - optimal_distance) / (max_distance - min_distance)
        
        # Height score (prefer table height)
        if height < min_height or height > max_height:
            height_score = 0
        else:
            height_score = 1.0 - abs(height - 0.1) / 0.5  # Prefer slightly above table
        
        # Approach direction bonus (eye-in-hand cameras see top surfaces better)
        approach_dir = grasp_pose[:3, 2]
        vertical_alignment = abs(approach_dir[2])  # How vertical is the approach
        vertical_bonus = min(1.0, vertical_alignment * 2)  # Bonus for vertical approaches
        
        return distance_score * height_score * vertical_bonus
    
    def _calculate_eyeinhand_combined_score(self, grasp_score, view_metrics, distance, 
                                           reachability, approach_angle):
        """Calculate combined score optimized for eye-in-hand setup"""
        # Adjusted weights for eye-in-hand
        weights = {
            'grasp_quality': 0.35,      # GraspNet score
            'reachability': 0.25,       # Robot reachability
            'view_quality': 0.20,       # Camera view quality
            'approach_angle': 0.15,     # Approach angle penalty
            'height_advantage': 0.05    # Camera height above object
        }
        
        # Calculate individual score components
        view_quality_score = view_metrics.get('view_quality', 1.0)
        approach_score = max(0, 1.0 - approach_angle / 90.0)  # Better for smaller angles
        height_score = min(1.0, max(0, view_metrics.get('height_advantage', 0) / 0.3))
        
        combined_score = (
            weights['grasp_quality'] * grasp_score +
            weights['reachability'] * reachability +
            weights['view_quality'] * view_quality_score +
            weights['approach_angle'] * approach_score +
            weights['height_advantage'] * height_score
        )
        
        return combined_score
    
    def _validate_eyeinhand_grasp(self, candidate):
        """Validate grasp candidate with eye-in-hand specific criteria"""
        # Standard validation
        if candidate.pose is None:
            return False
        
        if candidate.reachability_score < 0.2:  # Higher threshold for eye-in-hand
            return False
        
        if candidate.confidence < self.min_grasp_score:
            return False
        
        # Eye-in-hand specific validation
        if candidate.approach_angle > self.max_approach_angle:
            return False
        
        # Check if grasp is too high (eye-in-hand cameras might see background)
        if candidate.pose[2, 3] > 0.5:  # More than 50cm high
            return False
        
        return True
    
    def _rank_eyeinhand_grasps(self, candidates):
        """Rank grasps with eye-in-hand specific criteria"""
        # Sort by combined score
        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        # Apply stricter NMS for eye-in-hand (cameras close together)
        filtered = self._apply_nms(ranked, distance_threshold=0.03)  # 3cm threshold
        
        # Additional eye-in-hand specific filtering
        final_grasps = []
        for grasp in filtered:
            # Prefer grasps that are more perpendicular to surface
            if grasp.approach_angle < 45:  # Less than 45 degrees from vertical
                final_grasps.append(grasp)
        
        # If no good perpendicular grasps, keep best overall
        if not final_grasps and filtered:
            final_grasps = filtered[:5]
        
        return final_grasps
    
    def _visualize_eyeinhand_grasps(self, candidates, realsense_inputs, max_show=5):
        """Visualize grasps with eye-in-hand context"""
        geometries = []
        
        # Add world coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(world_frame)
        
        # Add table representation
        table = o3d.geometry.TriangleMesh.create_box(width=0.8, height=0.8, depth=0.02)
        table.translate([-0.4, -0.4, -0.01])
        table.paint_uniform_color([0.9, 0.9, 0.9])
        geometries.append(table)
        
        # Add camera positions
        for camera_id, input_data in realsense_inputs.items():
            camera_pos = np.array(input_data.camera_info["position"])
            camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            camera_sphere.translate(camera_pos)
            camera_sphere.paint_uniform_color([0.2, 0.2, 0.8])
            geometries.append(camera_sphere)
        
        # Add best grasps with quality indicators
        for i, candidate in enumerate(candidates[:max_show]):
            # Create grasp frame
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
            grasp_frame.transform(candidate.pose)
            
            # Color by quality and approach angle
            quality_color = min(1.0, candidate.score)
            approach_color = 1.0 - (candidate.approach_angle / 180.0)
            
            # Green for good quality, red for bad approach
            color = [1.0 - quality_color, quality_color * approach_color, 0]
            grasp_frame.paint_uniform_color(color)
            geometries.append(grasp_frame)
            
            # Add text info if possible
            rospy.loginfo(f"Grasp {i+1}: score={candidate.score:.3f}, "
                         f"approach={candidate.approach_angle:.1f}°, camera={candidate.camera_id}")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Eye-in-Hand Grasp Candidates"
        )
    
    # Utility methods (same as base class)
    def _pose_to_matrix(self, pose_vec):
        """Convert pose vector [x,y,z,qx,qy,qz,qw] to 4x4 transformation matrix"""
        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = pose_vec[:3]
        r = R.from_quat(pose_vec[3:])
        pose_matrix[:3, :3] = r.as_matrix()
        return pose_matrix
    
    def _transform_grasp_to_robot_frame(self, grasp_pose, camera_info):
        """Transform grasp pose from camera frame to robot base frame"""
        try:
            camera_position = np.array(camera_info["position"]).reshape(3, 1)
            camera_rotation = p.getMatrixFromQuaternion(camera_info["orientation"])
            camera_rotation = np.array(camera_rotation).reshape(3, 3)
            
            camera_to_robot = np.eye(4)
            camera_to_robot[:3, :3] = camera_rotation
            camera_to_robot[:3, 3] = camera_position.flatten()
            
            robot_grasp_pose = camera_to_robot @ grasp_pose
            return robot_grasp_pose
            
        except Exception as e:
            rospy.logerr(f"Error transforming grasp to robot frame: {str(e)}")
            return None
    
    def _apply_nms(self, candidates, distance_threshold=0.03):
        """Apply non-maximum suppression to remove similar grasps"""
        if not candidates:
            return []
        
        keep = [True] * len(candidates)
        
        for i in range(len(candidates)):
            if not keep[i]:
                continue
                
            for j in range(i + 1, len(candidates)):
                if not keep[j]:
                    continue
                
                pos_i = candidates[i].pose[:3, 3]
                pos_j = candidates[j].pose[:3, 3]
                
                if np.linalg.norm(pos_i - pos_j) < distance_threshold:
                    if candidates[i].score > candidates[j].score:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        
        return [candidates[i] for i in range(len(candidates)) if keep[i]]