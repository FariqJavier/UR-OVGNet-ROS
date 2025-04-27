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

GRASPNET_ROS_DIR = rospack.get_path('graspnet-ros') 
sys.path.append(os.path.join(GRASPNET_ROS_DIR, 'src'))

from graspnet_ros_utils.saving_image import create_and_publish_mask, save_color_and_depth_image
from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo as GraspCameraInfo, create_point_cloud_from_depth_image

class GraspNetNode:
    def __init__(self):
        rospy.init_node('ros_graspnet_node')
        color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        camera_info_sub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback)
        camera_status_sub = rospy.Subscriber('/start_motion', Bool, self.camera_status_callback)

        self.output_dir = rospy.get_param('~output_dir', 'doc/example_data')
        self.image_height = rospy.get_param('~image_height', 480.0)
        self.image_width = rospy.get_param('~image_width', 848.0)
        self.mask_margin_lr = rospy.get_param('~mask_margin_lr', 0.2)
        self.mask_margin_tb = rospy.get_param('~mask_margin_tb', 0.1)

        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.camera_info = None
        self.workspace_mask = None
        self.identifier = 1

    def camera_status_callback(self, msg):
        if self.latest_color is None or self.latest_depth is None or self.camera_info is None:
            rospy.logwarn("Waiting for color, depth, and camera info messages...")
            return
        
        try:
            if msg.data == True:
                self.identifier += 1

            if msg.data == False:
                self.identifier = 1

            self.workspace_mask = create_and_publish_mask(
                height=self.image_height,
                width=self.image_width,
                margin_lr=self.mask_margin_lr,
                margin_tb=self.mask_margin_tb,
                output_dir=self.output_dir,
                identifier=self.identifier
            )
            rospy.loginfo("Workspace mask created and published.")

            save_color_and_depth_image(
                color_msg=self.latest_color,
                depth_msg=self.latest_depth,
                camera_info_msg=self.camera_info,
                output_dir=self.output_dir,
                identifier=self.identifier,
                workspace_mask=self.workspace_mask
            )
            rospy.loginfo("Color and depth images saved.")
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
        GraspNetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass