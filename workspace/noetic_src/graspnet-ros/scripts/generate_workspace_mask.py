#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os
import numpy as np

def create_and_publish_mask():
    """
    Create a workspace mask with a rectangular ROI
    Args:
        height: Height of mask (default 270)
        width: Width of mask (default 480)
    Returns:
        Binary mask as numpy array
    """
    rospy.init_node('create_mask_node', anonymous=True)
    pub = rospy.Publisher('/new_mask', Image, queue_size=1)
    rate = rospy.Rate(1)  # Hz

    height = rospy.get_param('~image_height', 450)
    width = rospy.get_param('~image_width', 848)
    margin_lr = rospy.get_param('~margin_lr', 0.2)  # 10% margin from sides
    margin_tb = rospy.get_param('~margin_tb', 0.1)  # 10% margin from top/bottom
    output_path = rospy.get_param('~output_path', '/tmp/ros_images/new_workspace_mask.png')

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define workspace region (adjust these values based on your needs)
    margin_x = int(width * margin_lr)  # 10% margin from sides
    margin_y = int(height * margin_tb)  # 10% margin from top/bottom
    
    # Create rectangle ROI
    x1 = margin_x
    y1 = margin_y
    x2 = width - margin_x
    y2 = height - margin_y
    
    # Fill rectangle with white (255)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # Save resized mask (optional)
    cv2.imwrite(output_path, mask)
    rospy.loginfo(f"New mask saved to {output_path}")

    # Publish the mask
    bridge = CvBridge()
    ros_image = bridge.cv2_to_imgmsg(mask, encoding="mono8")

    while not rospy.is_shutdown():
        pub.publish(ros_image)
        rate.sleep()

if __name__ == '__main__':
    try:
        create_and_publish_mask()
    except rospy.ROSInterruptException:
        pass
