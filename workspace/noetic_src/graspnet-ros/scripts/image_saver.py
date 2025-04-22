#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import scipy.io as scio
import os
import numpy as np

def main():
    rospy.init_node('image_saver_node')
    bridge = CvBridge()

    rospy.loginfo("Waiting for color and depth images...")

    color_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
    depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw" , Image)
    camera_info_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)

    rospy.loginfo("Images received. Saving...")

    try:
        color_image = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)

        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        cx = camera_info_msg.K[2]
        cy = camera_info_msg.K[5]

        # Create output directory if not exists
        output_dir = rospy.get_param('~output_dir', '/tmp/ros_images')
        os.makedirs(output_dir, exist_ok=True)

        # Save images
        color_path = os.path.join(output_dir, 'color_ros.png')
        depth_path = os.path.join(output_dir, 'depth_ros.png')

        cv2.imwrite(color_path, color_image)

        # Graspnet needs depth in meters
        if depth_image.dtype == np.uint16:
            # Depth in millimeters (RealSense/ZED/others)
            depth_np = depth_image.astype(np.float32) / 1000.0 
        elif depth_image.dtype == np.float32:
            # Depth in meters
            depth_np = depth_image.copy()
        else:
            raise ValueError("Unsupported depth image type")

        np.save(os.path.join(output_dir, 'depth_ros.npy'), depth_np)  # Save as numpy array to preserve float values
        cv2.imwrite(depth_path, (depth_np * 1000).astype(np.uint16))  # Save visualization as PNG

        meta = {
            'intrinsic_matrix': np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ], dtype=np.float32),
            'factor_depth': 1.0  # Storing in millimeters
        }
        meta_path = os.path.join(output_dir, 'meta.mat')
        scio.savemat(meta_path, meta)

        rospy.loginfo(f"Color image saved to: {color_path}")
        rospy.loginfo(f"Depth image saved to: {depth_path}")
        rospy.loginfo(f"CameraInfo saved to: {meta_path}")

    except Exception as e:
        rospy.logerr(f"Failed to save images: {e}")

if __name__ == '__main__':
    main()
