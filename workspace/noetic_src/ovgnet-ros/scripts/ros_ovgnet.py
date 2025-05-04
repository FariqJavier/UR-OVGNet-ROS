#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import sys
import rospkg
import torch
import open3d as o3d
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from graspnetAPI import GraspGroup
from std_msgs.msg import Bool
from PIL import Image as PILImage

rospack = rospkg.RosPack()

OVGNET_ROS_DIR = rospack.get_path('ovgnet-ros')
sys.path.append(os.path.join(OVGNET_ROS_DIR, 'src'))

from ovgnet_ros_utils.processing_ovgnet import (
    get_realsense_input,
    get_groundingdino_inference
)

class OVGNetNode:
    def __init__(self):
        rospy.init_node('ros_graspnet_node')
        color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        camera_info_sub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback)
        camera_status_sub = rospy.Subscriber('/start_motion', Bool, self.camera_status_callback)

        self.output_dir = rospy.get_param('~output_dir', None)
        self.image_height = rospy.get_param('~image_height', 480.0)
        self.image_width = rospy.get_param('~image_width', 848.0)
        self.enable_color = rospy.get_param('~enable_color', True)
        self.enable_depth = rospy.get_param('~enable_depth', False)
        self.enable_camera_info = rospy.get_param('~enable_camera_info', False)
        self.use_mask = rospy.get_param('~use_mask', True)
        self.mask_margin_lr = rospy.get_param('~mask_margin_lr', 0.2)
        self.mask_margin_tb = rospy.get_param('~mask_margin_tb', 0.1)
        self.text_prompt = rospy.get_param("~text_prompt", None)
        self.config_path = rospy.get_param("~config_path", None)
        self.checkpoint_path = rospy.get_param("~checkpoint_path", None)
        self.box_threshold = rospy.get_param("~box_threshold", 0.3)
        self.text_threshold = rospy.get_param("~text_threshold", 0.25)
        self.token_spans = rospy.get_param("~token_spans", None)
        self.cpu_only = rospy.get_param("~cpu-only", False)
        os.makedirs(self.output_dir, exist_ok=True)
        self.identifier = 0

    def camera_status_callback(self, msg):
        if self.latest_color is None or self.latest_depth is None or self.camera_info is None:
            rospy.logwarn("Waiting for color, depth, and camera info messages ...")
            return
        if self.config_path is None or self.checkpoint_path is None or self.output_dir is None:
            rospy.logerr("Cannot find required path ...")
            return
        if self.text_prompt is None:
            rospy.logerr("Text prompt need to be specified ...")
        
        try:
            if msg.data == True:
                self.identifier += 1

            if msg.data == False:
                self.identifier = 1

            color_image_np, depth_image_np, camera_info = get_realsense_input (
                color_msg = self.latest_color,
                depth_msg = self.latest_depth,
                camera_info_msg = self.camera_info, 
                image_height = self.image_height,
                image_width = self.image_width,
                ws_margin_lr = self.mask_margin_lr,
                ws_margin_tb = self.mask_margin_tb,
                output_dir = os.path.join(self.output_dir, str(self.identifier)),
                identifier = str(self.identifier),
                enable_color = self.enable_color,
                enable_depth = self.enable_depth,
                enable_camera_info = self.camera_info,
                use_mask = self.use_mask
            )
            rospy.loginfo("Getting input from Intel Realsense D455 ...")

            box_filter, pred_label = get_groundingdino_inference (
                config_path = self.config_path,
                checkpoint_path = self.checkpoint_path,
                text_prompt = self.text_prompt,
                box_threshold = self.box_threshold,
                text_threshold = self.text_threshold,
                output_dir = os.path.join(self.output_dir, str(self.identifier)),
                color_image = color_image_np,
                token_spans = self.token_spans,
                cpu_only = self.cpu_only
            )
            rospy.loginfo("Getting groundingdino inference")
        except Exception as e:
            rospy.logerr(f"Failed to save images: {e}")
            return

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def color_callback(self, msg):
        self.latest_color = msg

    def depth_callback(self, msg):
        self.latest_depth = msg


if __name__ == '__main__':
    try:
        OVGNetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass