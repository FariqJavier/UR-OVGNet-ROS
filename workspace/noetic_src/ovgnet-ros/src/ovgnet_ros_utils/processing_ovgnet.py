import rospy
import os
import json
import sensor_msgs
import numpy as np
import open3d as o3d
from typing import Union
from torch import Tensor
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
# import tf2_geometry_msgs

from groundingdino_ros_utils.processing_groundingdino import (
    load_image,
    load_model,
    get_grounding_output,
    plot_boxes_to_image,
)

from realsense_ros_utils.saving_image import (
    create_and_publish_mask,
    get_color_and_depth_image
)

from graspnet_ros_utils.grasp_detector import Graspnet
from graspnet_ros_utils.multiview_grasp_planner import MultiViewGraspPlanner
from graspnet_ros_utils.processing_graspnet import (
    get_fuse_pointcloud,
    get_single_pointcloud
)

def get_realsense_input (
    color_msg: sensor_msgs.msg.Image, 
    depth_msg: sensor_msgs.msg.Image, 
    camera_info_msg: sensor_msgs.msg.CameraInfo, 
    image_height: int = 480,
    image_width: int = 848,
    ws_margin_lr: float = 0.1,  # 10% margin from sides
    ws_margin_tb: float = 0.1,  # 10% margin from top/bottom
    output_dir: str = '/tmp/ros_images',
    identifier: str = '1',
    enable_color: bool = True,
    enable_depth: bool = False,
    enable_camera_info: bool = False,
    use_mask: bool = True
    ) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[any, None]]:
    """
    Create workspace mask and capture realsense input every robot detecting motion sequence.
    Args:
        image_height (int): The height of the image captured by realsense
        image_width (int): The width of the image captured by realsense
        ws_margin_lr (float): The left and right margin used for creating workspace mask
        ws_margin_tb (float): The top and bottom margin used for creating workspace mask
        output_dir (str): Directory for saving the captured images
        identifier (str): Unique identifier for the image set.
        enable_color (bool): Flag for getting colored realsense input
        enable_depth (bool): Flag for getting depth relasense input
        enable_camera_info (bool): Flag for getting relasense intrinsic
        use_mask (bool): Flag for using masked processed realsense input instead of raw realsense input
    """
    try:
        ws_mask = create_and_publish_mask(
            height=image_height,
            width=image_width,
            margin_lr=ws_margin_lr,
            margin_tb=ws_margin_tb,
            output_dir=output_dir,
            identifier=identifier
        )

        color_image, depth_image, camera_info = get_color_and_depth_image (
            color_msg=color_msg,
            depth_msg=depth_msg,
            camera_info_msg=camera_info_msg,
            output_dir=output_dir,
            identifier=identifier,
            workspace_mask=ws_mask,
            enable_color=enable_color,
            enable_depth=enable_depth,
            enable_camera_info=enable_camera_info,
            use_mask=use_mask
        )

        return color_image, depth_image, camera_info
    except Exception as e:
        raise RuntimeError(f"Failed to get realsense input: {e}")

def get_groundingdino_inference (
    config_path: str,
    checkpoint_path: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    output_dir: str,
    color_image: np.ndarray,
    token_spans: str = None,
    cpu_only: bool = False
    ) -> tuple[Union[sensor_msgs.msg.Image, None], Union[sensor_msgs.msg.Image, None]]:
    """
    Doing inference on input image
    Args:
        config_path (str): Path of the groundingdino config
        checkpoint_path (str): Path of the groundingdino weight
        text_prompt (str):
        box_threshold (float): 
        text_threshold (float):
        token_spans (str): 
        output_dir (str):
        color_image (np.ndarray):
        cpu_only (bool):
    """
    try:
        model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            cpu_only=cpu_only
        )

        # Load the image using PIL
        image_pil, image_tensor = load_image(color_image)

        # visualize raw image
        image_pil.save(os.path.join(output_dir, "input_groundingdino.png"))
            
        # Return None, None if the failed to get groundingdino inference    
        box_filter, pred_label = get_grounding_output(
            model=model, image=image_tensor, caption=text_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold, with_logits=True, cpu_only=cpu_only, token_spans=token_spans
        )
        if box_filter is None:
            return None, None

        # Draw results
        size = image_pil.size
        output_image, mask = plot_boxes_to_image(image_pil, {
            "boxes": box_filter,
            "labels": pred_label,
            "size": [size[1], size[0]],  # H,W
        })
        output_path = os.path.join(output_dir, "result_groundingdino.jpg")
        output_image.save(output_path)
        # rospy.loginfo(f"Inference complete. Result saved to {output_path}")

        return box_filter, pred_label
    except Exception as e:
        raise RuntimeError(f"Failed to get groundingdino inference: {e}")

def get_graspnet_inference (
    checkpoint_path: str,
    refine_approach_dist: float,
    dist_thresh: float,
    angle_thresh: int,
    mask_thresh: float,
    realsense_input_dict: any,
    groundingdino_output_dict: any,
    output_dir: str,
    frame_id: int,
    visualize: bool = False
    ):
    """
    Run GraspNet prediction on the point cloud and save visualizations
    
    Args:
        checkpoint_path: Path to the GraspNet checkpoint
        refine_approach_dist: Distance for approach refinement (meters)
        dist_thresh: Distance threshold for grasp assignment (meters)
        angle_thresh: Angle threshold for grasp filtering (degrees)
        mask_thresh: Minimum number of grasps required
        realsense_input_dict: Dictionary with realsense data
        groundingdino_output_dict: Dictionary with grounding DINO output
        output_dir: Directory to save output files
        frame_id: Current frame ID
        visualize: Whether to show visualizations (default: False)
    
    Returns:
        tuple: (fuse_pcd, grasp_poses, scores) - point cloud, grasp poses and their scores
    """
    try:
        # Initialize Graspnet
        graspnet = Graspnet(
            checkpoint_path=checkpoint_path,
            refine_approach_dist=refine_approach_dist,
            dist_thresh=dist_thresh,
            angle_thresh=angle_thresh,
            mask_thresh=mask_thresh
        )

        # Get pointcloud from input image
        fused_pcd_world, fused_pcd_canonical, fused_trans_world, fused_trans_canonical = get_fuse_pointcloud(
            realsense_input_dict=realsense_input_dict,
            groundingdino_output_dict=groundingdino_output_dict,
            frame_id=frame_id
        )

        # # Visualize the fused point cloud
        # o3d.visualization.draw_geometries([fuse_pcd])

        # Save the fused point cloud to a file
        fused_pcd_world_path = os.path.join(output_dir, f"fused_point_cloud__world_{str(frame_id)}.pcd")
        o3d.io.write_point_cloud(fused_pcd_world_path, fused_pcd_world)
        rospy.loginfo(f"Fused point cloud saved to: {fused_pcd_world_path}")

        fused_pcd_canonical_path = os.path.join(output_dir, f"fused_point_cloud_canonical_{str(frame_id)}.pcd")
        o3d.io.write_point_cloud(fused_pcd_canonical_path, fused_pcd_canonical)
        rospy.loginfo(f"Fused point cloud saved to: {fused_pcd_canonical_path}")

        # Generate grasp pose from graspnet
        grasp_poses, geometries, scores = graspnet.grasp_detection_real_world(
            fused_pcd_world,
            fused_pcd_canonical, 
            fused_trans_canonical,
            get_visual=True, 
            top_down_only=True
        )

        # Save grasp data as JSON
        grasp_data = []
        for i, (pose, score) in enumerate(zip(grasp_poses, scores)):
            grasp_data.append({
                "id": i,
                "position": [float(p) for p in pose[:3]],
                "orientation": [float(q) for q in pose[3:7]],
                "score": float(score)
            })
        
        with open(os.path.join(output_dir, f"grasp_data_{str(frame_id)}.json"), 'w') as f:
            json.dump(grasp_data, f, indent=2)

        # Log grasp results
        if len(grasp_poses) > 0:
            rospy.loginfo(f"Found {len(grasp_poses)} valid grasps")
            best_score = scores[0]
            best_pose = grasp_poses[0]
            # rospy.loginfo(f"Best grasp score: {best_score:.4f}")
            # rospy.loginfo(f"Best grasp position: [{best_pose[0]:.4f}, {best_pose[1]:.4f}, {best_pose[2]:.4f}]")
        else:
            rospy.logerr("No valid grasp poses found")
            return [], [], []

        # Create visualization outputs
        # 1. Save individual grasps meshes
        grasp_dir = os.path.join(output_dir, f"grasps_{str(frame_id)}")
        os.makedirs(grasp_dir, exist_ok=True)
        
        for i, geom in enumerate(geometries):
            grasp_path = os.path.join(grasp_dir, f"grasp_{i}_score_{scores[i]:.4f}.ply")
            o3d.io.write_triangle_mesh(grasp_path, geom)
        
        # 2. Save combined grasp visualization
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Color the grasps by their score (red=low, green=high)
        for i, geom in enumerate(geometries):
            # Normalize score to 0-1
            normalized_score = scores[i]
            # Create color mapping (red to green based on score)
            color = np.array([1.0 - normalized_score, normalized_score, 0.0])
            # Apply color to the mesh
            geom.paint_uniform_color(color)
        
        # Combine all grasp geometries into one mesh
        combined_grasps = o3d.geometry.TriangleMesh()
        for geom in geometries:
            combined_grasps += geom
        
        # Save combined grasps
        combined_path = os.path.join(output_dir, f"combined_grasps_{str(frame_id)}.ply")
        o3d.io.write_triangle_mesh(combined_path, combined_grasps)
        rospy.loginfo(f"Saved combined grasps to {combined_path}")
        
        # 3. Create a full scene visualization with the best grasp highlighted
        if len(geometries) > 0:
            # Highlight the best grasp in blue
            geometries[0].paint_uniform_color([0.0, 0.5, 1.0])  # Blue for best grasp
            
            # Create a screenshot of the scene with the best grasp
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=visualize)
            vis.add_geometry(fused_pcd_world)
            vis.add_geometry(coord_frame)
            
            # Add all grasps
            for geom in geometries:
                vis.add_geometry(geom)
                
            # Set view
            view_control = vis.get_view_control()
            view_control.set_front([0, 0, -1])  # View from front
            view_control.set_up([0, -1, 0])     # Up direction
            view_control.set_zoom(0.7)
            
            # Update visualization and capture the image
            vis.poll_events()
            vis.update_renderer()
            
            # Save image
            image_path = os.path.join(output_dir, f"grasp_scene_{str(frame_id)}.png")
            vis.capture_screen_image(image_path)
            rospy.loginfo(f"Saved visualization image to {image_path}")
            
            # If visualize is True, show the window
            if visualize:
                rospy.loginfo("Showing visualization window. Close the window to continue.")
                vis.run()
            
            vis.destroy_window()
        
        # Return both the fuse point cloud, best grasp pose, and best grasp score
        return fused_pcd_world, best_pose, best_score

    except Exception as e:
        raise RuntimeError(f"Failed to get graspnet inference: {e}")

def get_graspnet_inference_on_multiview (
    checkpoint_path: str,
    refine_approach_dist: float,
    dist_thresh: float,
    angle_thresh: int,
    mask_thresh: float,
    realsense_input_dict: any,
    groundingdino_output_dict: any,
    output_dir: str,
    frame_id: int,
    visualize: bool = False
    ):
    """
    Run GraspNet prediction on the point cloud and save visualizations
    
    Args:
        checkpoint_path: Path to the GraspNet checkpoint
        refine_approach_dist: Distance for approach refinement (meters)
        dist_thresh: Distance threshold for grasp assignment (meters)
        angle_thresh: Angle threshold for grasp filtering (degrees)
        mask_thresh: Minimum number of grasps required
        realsense_input_dict: Dictionary with realsense data
        groundingdino_output_dict: Dictionary with grounding DINO output
        output_dir: Directory to save output files
        frame_id: Current frame ID
        visualize: Whether to show visualizations (default: False)
    
    Returns:
        tuple: (fuse_pcd, grasp_poses, scores) - point cloud, grasp poses and their scores
    """
    try:
        # Initialize Graspnet
        graspnet = Graspnet(
            checkpoint_path=checkpoint_path,
            refine_approach_dist=refine_approach_dist,
            dist_thresh=dist_thresh,
            angle_thresh=angle_thresh,
            mask_thresh=mask_thresh
        )

        # Initialize the multi-view planner
        planner = MultiViewGraspPlanner(graspnet, robot_base_frame='base_link')

        # Generate grasps from multiple views
        grasp_candidates = planner.plan_multiview_grasps(
            realsense_inputs=realsense_input_dict,
            detection_results=groundingdino_output_dict,
            get_visual=True  # For debugging
        )

        # Save grasp data as JSON
        grasp_data = []
        for i, grasp in enumerate(grasp_candidates):
            pose_matrix = grasp.pose  # This is a 4x4 matrix
            
            # Extract translation
            position = pose_matrix[:3, 3]

            # Extract rotation matrix and convert to quaternion
            rotation = pose_matrix[:3, :3]
            quat = R.from_matrix(rotation).as_quat()  # returns [x, y, z, w]

            grasp_data.append({
                "id": i,
                "position": position.tolist(),
                "orientation": quat.tolist(),
                "score": float(grasp.score),
                "confidence": float(grasp.confidence),
                "distance_to_robot": float(grasp.distance_to_robot),
                "view_angle": float(grasp.view_angle),
                "reachability_score": float(grasp.reachability_score)
            })
        
        with open(os.path.join(output_dir, f"grasp_data_{str(frame_id)}.json"), 'w') as f:
            json.dump(grasp_data, f, indent=2)

        # Log grasp results
        if len(grasp_candidates) > 0:
            rospy.loginfo(f"Found {len(grasp_candidates)} valid grasps")
            best_pose = grasp_candidates[0]
            best_score = grasp_candidates[0].score
            best_confidence = grasp_candidates[0].confidence
            best_distance_to_robot = grasp_candidates[0].distance_to_robot
            best_view_angle = grasp_candidates[0].view_angle
            best_reachability_score = grasp_candidates[0].reachability_score
            # rospy.loginfo(f"Best grasp score: {best_score:.4f}")
            # rospy.loginfo(f"Best grasp position: [{best_pose[0]:.4f}, {best_pose[1]:.4f}, {best_pose[2]:.4f}]")
            # rospy.loginfo(f"Best grasp Confidence: {best_confidence:.3f}")
            # rospy.loginfo(f"Best grasp Distance: {best_distance:.3f} m")
            # rospy.loginfo(f"Best grasp  View angle: {best_angle:.1f} degrees")
            # rospy.loginfo(f"Best grasp  Reachability: {best_reachability_score:.1f}")
            
        else:
            rospy.logerr("No valid grasp poses found")
            return [], [], [], [], [], []

        # Create visualization outputs
        # 1. Save individual grasps meshes
        grasp_dir = os.path.join(output_dir, f"grasps_{str(frame_id)}")
        os.makedirs(grasp_dir, exist_ok=True)
        
        for i, g in enumerate(grasp_candidates):
            grasp_path = os.path.join(grasp_dir, f"grasp_{i}_score_{g.score:.4f}.ply")
            o3d.io.write_triangle_mesh(grasp_path, g.geometry)

            # Normalize score to 0-1
            normalized_score = g.score[i]
            # Create color mapping (red to green based on score)
            color = np.array([1.0 - normalized_score, normalized_score, 0.0])
            # Apply color to the mesh
            g.geometry.paint_uniform_color(color)

        
        # 2. Save combined grasp visualization
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Combine all grasp geometries into one mesh
        combined_grasps = o3d.geometry.TriangleMesh()
        for grasp in grasp_candidates:
            combined_grasps += grasp.geometry
        
        # Save combined grasps
        combined_path = os.path.join(output_dir, f"combined_grasps_{str(frame_id)}.ply")
        o3d.io.write_triangle_mesh(combined_path, combined_grasps)
        rospy.loginfo(f"Saved combined grasps to {combined_path}")
        
        # Return both the best grasp pose with its score, confidence, distance, view angle, and reachability score
        return best_pose, best_score, best_confidence, best_distance_to_robot, best_view_angle, best_reachability_score

    except Exception as e:
        raise RuntimeError(f"Failed to get graspnet inference: {e}")

# def create_pose_msg(
#     grasp_pose: np.ndarray, 
#     frame_id: str = "world"
#     ) -> geometry_msgs.msg.PoseStamped:
#     """
#     Creates a geometry_msgs.msg.PoseStamped object.

#     Args:
#         grasp_pose: List representation of 6DoF grasp pose
#         frame_id: Frame reference of the pose

#     Return:
#         PoseStamped object of the grasp pose
#     """
#     if not grasp_pose:
#         return [] 

#     pose_stamped = geometry_msgs.msg.PoseStamped()
#     pose_stamped.header.stamp = rospy.Time.now() # Or get time from appropriate source
#     pose_stamped.header.frame_id = frame_id # Frame this pose is defined in
#     pose_stamped.pose.position.x = grasp_pose[0]
#     pose_stamped.pose.position.y = grasp_pose[1]
#     pose_stamped.pose.position.z = grasp_pose[2]

#     pose_stamped.pose.orientation.x = grasp_pose[3]
#     pose_stamped.pose.orientation.y = grasp_pose[4]
#     pose_stamped.pose.orientation.z = grasp_pose[5]
#     pose_stamped.pose.orientation.w = grasp_pose[6]

#     return pose_stamped

def create_pose_msg(
    grasp_pose: np.ndarray, 
    frame_id: str = "world"
    ) -> geometry_msgs.msg.PoseStamped:
    """
    Creates a geometry_msgs.msg.PoseStamped object from grasp pose.

    Args:
        grasp_pose: Either a 4x4 transformation matrix or 7-element array [x,y,z,qx,qy,qz,qw]
        frame_id: Frame reference of the pose

    Return:
        PoseStamped object of the grasp pose
    """
    # Check if input is valid
    if grasp_pose is None:
        rospy.logerr("Grasp pose is None")
        return None
    
    # Handle empty list/array
    if isinstance(grasp_pose, (list, np.ndarray)) and len(grasp_pose) == 0:
        rospy.logerr("Grasp pose is empty")
        return None

    pose_stamped = geometry_msgs.msg.PoseStamped()
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.header.frame_id = frame_id
    
    # Handle different input formats
    if isinstance(grasp_pose, np.ndarray) and grasp_pose.shape == (4, 4):
        # Input is a 4x4 transformation matrix
        position = grasp_pose[:3, 3]
        rotation_matrix = grasp_pose[:3, :3]
        
        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # Returns [x, y, z, w]
        
        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]
        
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]
        
    elif isinstance(grasp_pose, (list, np.ndarray)) and len(grasp_pose) == 7:
        # Input is [x, y, z, qx, qy, qz, qw]
        pose_stamped.pose.position.x = float(grasp_pose[0])
        pose_stamped.pose.position.y = float(grasp_pose[1])
        pose_stamped.pose.position.z = float(grasp_pose[2])
        
        pose_stamped.pose.orientation.x = float(grasp_pose[3])
        pose_stamped.pose.orientation.y = float(grasp_pose[4])
        pose_stamped.pose.orientation.z = float(grasp_pose[5])
        pose_stamped.pose.orientation.w = float(grasp_pose[6])
        
    else:
        rospy.logerr(f"Invalid grasp pose format: {type(grasp_pose)} with shape/length {getattr(grasp_pose, 'shape', len(grasp_pose) if hasattr(grasp_pose, '__len__') else 'N/A')}")
        return None
    
    # Validate quaternion (should be normalized)
    quat_norm = np.sqrt(pose_stamped.pose.orientation.x**2 + 
                       pose_stamped.pose.orientation.y**2 + 
                       pose_stamped.pose.orientation.z**2 + 
                       pose_stamped.pose.orientation.w**2)
    
    if abs(quat_norm - 1.0) > 0.01:  # Allow small numerical errors
        rospy.logwarn(f"Quaternion not normalized (norm={quat_norm:.4f}). Normalizing...")
        pose_stamped.pose.orientation.x /= quat_norm
        pose_stamped.pose.orientation.y /= quat_norm
        pose_stamped.pose.orientation.z /= quat_norm
        pose_stamped.pose.orientation.w /= quat_norm

    return pose_stamped