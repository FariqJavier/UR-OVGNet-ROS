import rospy
import os
import sensor_msgs
import numpy as np
from typing import Union
from torch import Tensor

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
from graspnet_ros_utils.processing_graspnet import get_single_fuse_pointcloud

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
        rospy.loginfo(f"Inference complete. Result saved to {output_path}")

        return box_filter, pred_label
    except Exception as e:
        raise RuntimeError(f"Failed to get groundingdino inference: {e}")

def get_graspnet_inference (
    checkpoint_path: str,
    refine_approach_dist: float,
    dist_thresh: float,
    angle_thresh: int,
    mask_thresh: float,
    color_image: np.ndarray,
    depth_image: np.ndarray,
    camera_info: any,
    box_filter: Tensor
    ):
    """
    Run GraspNet prediction on the point cloud
    
    Args:
        pcd: Open3D point cloud
        graspnet_config: Configuration for GraspNet
        box_filter (Tensor(0,4)): The first index of the bounding box generated from groundingdino
    
    Returns:
        grasp_poses: Predicted grasp poses
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
        pcd = get_single_fuse_pointcloud(
            camera_info=camera_info,
            color_image_np=color_image,
            depth_image_np=depth_image,
            box_filter=box_filter,
        )

    except Exception as e:
        raise RuntimeError(f"Failed to get graspnet inference: {e}")

    

# def save_graspnet_inference ():


# def calculate_motion_planning ():