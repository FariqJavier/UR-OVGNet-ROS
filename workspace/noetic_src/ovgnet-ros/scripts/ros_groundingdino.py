#!/usr/bin/env python3
import rospy
import rospkg
from cv_bridge import CvBridge
import os
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np

rospack = rospkg.RosPack()

OVGNET_ROS_DIR = rospack.get_path('ovgnet-ros')
sys.path.append(os.path.join(OVGNET_ROS_DIR, 'src'))

from groundingdino_ros_utils.processing_groundingdino import (
    load_image,
    load_model,
    get_grounding_output,
    plot_boxes_to_image,
)
from realsense_ros_utils.saving_image import load_color_and_depth_image

class GroundingDinoNode:
    def __init__(self):
        rospy.init_node("grounding_dino_node")

        self.text_prompt = rospy.get_param("~text_prompt", "a bottle")
        self.config_path = rospy.get_param("~config_path")
        self.checkpoint_path = rospy.get_param("~checkpoint_path")
        self.box_threshold = rospy.get_param("~box_threshold", 0.3)
        self.text_threshold = rospy.get_param("~text_threshold", 0.25)
        self.token_spans = rospy.get_param("~token_spans", None)
        self.cpu_only = rospy.get_param("~cpu-only", False)
        self.output_dir = rospy.get_param("~output_dir", "/tmp/dino_ros_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.input_dir = rospy.get_param("~input_dir", "/tmp/dino_ros_input")

        # Load GroundingDINO model
        rospy.loginfo("Loading GroundingDino model ...")
        self.model = load_model(self.config_path, self.checkpoint_path, self.cpu_only)

        # Process all subdirectories
        self.get_image_files()

    def get_image_files(self):
        """Process all numbered subdirectories (1-13) in the input directory."""
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"Input directory '{self.input_dir}' not found.")
        
        # Process each subdirectory in order
        for subdir_num in range(1, 13):  # 1 to 12
            subdir_path = os.path.join(self.input_dir, str(subdir_num))
            try:
                # Check if the subdirectory exists
                if not os.path.isdir(subdir_path):
                    raise FileNotFoundError(f"Subdirectory '{subdir_num}' not found in '{self.input_dir}'")

                rospy.loginfo(f"Processing subdirectory: {subdir_path}")
                
                color_image_path, _, _ = load_color_and_depth_image(
                    subdir_path, str(subdir_num), enable_color=True, enable_depth=False, enable_camera_info=False, use_mask=True
                )
                    
                self.process_image_file(color_image_path, subdir_path)
            except Exception as e:
                rospy.logerr(f"Error processing subdirectory '{subdir_num}': {e}")

    def process_image_file(self, image_path, output_dir):
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file '{self.config_path}' not found.")
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file '{self.checkpoint_path}' not found.")
            
            if self.token_spans is not None:
                self.text_threshold = None
                print("Using token_spans. Set the text_threshold to None.")

            rospy.loginfo(f"Processing subdirectory: {image_path}")

            # Load the image using PIL
            image_pil, image_tensor = load_image(image_path)

            # visualize raw image
            image_pil.save(os.path.join(output_dir, "input_groundingdino.png"))
            
            boxes, labels = get_grounding_output(
                model=self.model, image=image_tensor, caption=self.text_prompt,
                box_threshold=self.box_threshold, text_threshold=self.text_threshold, with_logits=True, cpu_only=self.cpu_only, token_spans=None
            )

            # Draw results
            size = image_pil.size
            output_image, mask = plot_boxes_to_image(image_pil, {
                "boxes": boxes,
                "labels": labels,
                "size": [size[1], size[0]],  # H,W
            })
            output_path = os.path.join(output_dir, "result_groundingdino.jpg")
            output_image.save(output_path)
            rospy.loginfo(f"Inference complete. Result saved to {output_path}")

        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")

if __name__ == "__main__":
    try:
        GroundingDinoNode()
    except rospy.ROSInterruptException:
        pass
