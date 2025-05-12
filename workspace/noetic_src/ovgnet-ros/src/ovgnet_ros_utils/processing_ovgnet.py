import rospy
import os
import json
import sensor_msgs
import numpy as np
import open3d as o3d
from typing import Union
from torch import Tensor
import geometry_msgs.msg

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
from graspnet_ros_utils.processing_graspnet import get_fuse_pointcloud

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
        fuse_pcd = get_fuse_pointcloud(
            realsense_input_dict=realsense_input_dict,
            groundingdino_output_dict=groundingdino_output_dict,
            frame_id=frame_id
        )

        # # Visualize the fused point cloud
        # o3d.visualization.draw_geometries([fuse_pcd])

        # Save the fused point cloud to a file
        fused_pcd_path = os.path.join(output_dir, f"fused_point_cloud_{str(frame_id)}.pcd")
        o3d.io.write_point_cloud(fused_pcd_path, fuse_pcd)
        rospy.loginfo(f"Fused point cloud saved to: {fused_pcd_path}")

        # Generate grasp pose from graspnet
        grasp_poses, geometries, scores = graspnet.grasp_detection_real_world(
            fuse_pcd, 
            get_visual=False, 
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
            return fuse_pcd, [], []

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
            vis.add_geometry(fuse_pcd)
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
        return fuse_pcd, best_pose, best_score

    except Exception as e:
        raise RuntimeError(f"Failed to get graspnet inference: {e}")

def create_pose_msg(
    grasp_pose: np.ndarray, 
    frame_id: str = "world"
    ) -> geometry_msgs.msg.PoseStamped:
    """
    Creates a geometry_msgs.msg.PoseStamped object.

    Args:
        grasp_pose: List representation of 6DoF grasp pose
        frame_id: Frame reference of the pose

    Return:
        PoseStamped object of the grasp pose
    """
    pose_stamped = geometry_msgs.msg.PoseStamped()
    pose_stamped.header.stamp = rospy.Time.now() # Or get time from appropriate source
    pose_stamped.header.frame_id = frame_id # Frame this pose is defined in
    pose_stamped.pose.position.x = grasp_pose[0]
    pose_stamped.pose.position.y = grasp_pose[1]
    pose_stamped.pose.position.z = grasp_pose[2]

    pose_stamped.pose.orientation.x = grasp_pose[3]
    pose_stamped.pose.orientation.y = grasp_pose[4]
    pose_stamped.pose.orientation.z = grasp_pose[5]
    pose_stamped.pose.orientation.w = grasp_pose[6]

    return pose_stamped