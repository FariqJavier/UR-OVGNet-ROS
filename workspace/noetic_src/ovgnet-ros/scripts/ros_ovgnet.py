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
import threading
import queue
import time
import copy

rospack = rospkg.RosPack()

OVGNET_ROS_DIR = rospack.get_path('ovgnet-ros')
sys.path.append(os.path.join(OVGNET_ROS_DIR, 'src'))

from ovgnet_ros_utils.processing_ovgnet import (
    get_realsense_input,
    get_groundingdino_inference,
    get_graspnet_inference
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
        self.enable_depth = rospy.get_param('~enable_depth', True)
        self.enable_camera_info = rospy.get_param('~enable_camera_info', True)
        self.use_mask = rospy.get_param('~use_mask', True)
        self.mask_margin_lr = rospy.get_param('~mask_margin_lr', 0.2)
        self.mask_margin_tb = rospy.get_param('~mask_margin_tb', 0.1)
        self.groundingdino_text_prompt = rospy.get_param("~groundingdino_text_prompt", None)
        self.groundingdino_config_path = rospy.get_param("~groundingdino_config_path", None)
        self.groundingdino_checkpoint_path = rospy.get_param("~groundingdino_checkpoint_path", None)
        self.groundingdino_box_threshold = rospy.get_param("~groundingdino_box_threshold", 0.3)
        self.groundingdino_text_threshold = rospy.get_param("~groundingdino_text_threshold", 0.25)
        self.groundingdino_token_spans = rospy.get_param("~groundingdino_token_spans", None)
        self.groundingdino_cpu_only = rospy.get_param("~groundingdino_cpu_only", False)
        self.graspnet_checkpoint_path = rospy.get_param("~graspnet_checkpoint_path", None)
        self.grapnet_refine_approach_dist = rospy.get_param("~grapnet_refine_approach_dist", 0.01)
        self.graspnet_dist_thresh = rospy.get_param("~graspnet_dist_thresh", 0.05)
        self.graspnet_angle_thresh = rospy.get_param("~graspnet_angle_thresh", 15)
        self.graspnet_mask_thresh = rospy.get_param("~graspnet_mask_thresh", 0.5)
        os.makedirs(self.output_dir, exist_ok=True)
        self.identifier = 0

        # Check if GPU is available
        self.use_gpu = torch.cuda.is_available() and not self.groundingdino_cpu_only
        if self.use_gpu:
            device_count = torch.cuda.device_count()
            self.device = torch.device('cuda:0')
            rospy.loginfo(f"Using GPU acceleration. Available devices: {device_count}")
        else:
            self.device = torch.device('cpu')
            rospy.loginfo("Using CPU for inference")

        # Create processing queue and worker thread
        self.processing_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory buildup
        self.worker_thread = threading.Thread(target=self.processing_worker)
        self.worker_thread.daemon = True  # Thread will exit when main program exits
        self.worker_thread.start()
        
        # Add a rate limiter to prevent overwhelming the queue
        self.last_enqueue_time = 0
        self.min_enqueue_interval = 1.0  # Minimum seconds between enqueueing new frames

    # def camera_status_callback(self, msg):
    #     if self.latest_color is None or self.latest_depth is None or self.camera_info is None:
    #         rospy.logwarn("Waiting for color, depth, and camera info messages ...")
    #         return
    #     if self.config_path is None or self.checkpoint_path is None or self.output_dir is None:
    #         rospy.logerr("Cannot find required path ...")
    #         return
    #     if self.text_prompt is None:
    #         rospy.logerr("Text prompt need to be specified ...")
        
    #     try:
    #         if msg.data == True:
    #             self.identifier += 1

    #         if msg.data == False:
    #             self.identifier = 0
    #             rospy.loginfo("Processing image finished")
    #             return

    #         color_image_np, depth_image_np, camera_info = get_realsense_input (
    #             color_msg = self.latest_color,
    #             depth_msg = self.latest_depth,
    #             camera_info_msg = self.camera_info, 
    #             image_height = self.image_height,
    #             image_width = self.image_width,
    #             ws_margin_lr = self.mask_margin_lr,
    #             ws_margin_tb = self.mask_margin_tb,
    #             output_dir = os.path.join(self.output_dir, str(self.identifier)),
    #             identifier = str(self.identifier),
    #             enable_color = self.enable_color,
    #             enable_depth = self.enable_depth,
    #             enable_camera_info = self.camera_info,
    #             use_mask = self.use_mask
    #         )
    #         rospy.loginfo("Getting input from Intel Realsense D455 ...")

    #         box_filter, pred_label = get_groundingdino_inference (
    #             config_path = self.config_path,
    #             checkpoint_path = self.checkpoint_path,
    #             text_prompt = self.text_prompt,
    #             box_threshold = self.box_threshold,
    #             text_threshold = self.text_threshold,
    #             output_dir = os.path.join(self.output_dir, str(self.identifier)),
    #             color_image = color_image_np,
    #             token_spans = self.token_spans,
    #             cpu_only = self.cpu_only
    #         )
    #         rospy.loginfo("Getting groundingdino inference")
    #     except Exception as e:
    #         rospy.logerr(f"Failed to save images: {e}")
    #         return

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def color_callback(self, msg):
        self.latest_color = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def camera_status_callback(self, msg):
        # Skip processing if we don't have all the data yet
        if self.latest_color is None or self.latest_depth is None or self.camera_info is None:
            rospy.logwarn("Waiting for color, depth, and camera info messages ...")
            return
            
        # Skip processing if required paths are missing
        if self.groundingdino_config_path is None or self.groundingdino_checkpoint_path is None or self.output_dir is None or self.graspnet_checkpoint_path is None:
            rospy.logerr("Cannot find required path ...")
            return
            
        # Skip processing if text prompt is missing
        if self.groundingdino_text_prompt is None:
            rospy.logerr("Text prompt need to be specified ...")
            return
        
        # Handle the message
        if msg.data == True:
            # Check if queue is full
            if self.processing_queue.full():
                rospy.logwarn("Processing queue is full, skipping frame")
                return
                
            # Implement rate limiting
            current_time = time.time()
            if current_time - self.last_enqueue_time < self.min_enqueue_interval:
                rospy.loginfo("Rate limiting: skipping this frame")
                return
                
            self.identifier += 1
            
            # Create copies of the messages
            color_msg_copy = copy.deepcopy(self.latest_color)
            depth_msg_copy = copy.deepcopy(self.latest_depth)
            camera_info_copy = copy.deepcopy(self.camera_info)
            
            # Add to processing queue
            try:
                self.processing_queue.put((self.identifier, color_msg_copy, depth_msg_copy, camera_info_copy), 
                                         block=False)  # Non-blocking to avoid hanging if queue is full
                self.last_enqueue_time = current_time
                rospy.loginfo(f"Enqueued frame {self.identifier} for processing")
            except queue.Full:
                rospy.logwarn("Queue is full, dropping frame")
            
        elif msg.data == False:
            self.identifier = 0
            rospy.loginfo("Processing image finished")
            return

    def processing_worker(self):
        """Worker function that runs in a separate thread to process frames"""
        while not rospy.is_shutdown():
            try:
                # Get task from queue, blocks until an item is available
                task = self.processing_queue.get(timeout=1.0)
                
                # Unpack the task data
                identifier, color_msg, depth_msg, camera_info_msg = task
                
                # Process the frame (moved from camera_status_callback)
                self.process_frame(identifier, color_msg, depth_msg, camera_info_msg)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                # Timeout on queue.get, just continue the loop
                continue
            except Exception as e:
                rospy.logerr(f"Error in worker thread: {e}")

    def process_frame(self, identifier, color_msg, depth_msg, camera_info_msg):
        """Process a single frame with all the detection logic"""
        try:
            rospy.loginfo(f"Processing frame {identifier}")
            
            color_image_np, depth_image_np, camera_info = get_realsense_input(
                color_msg=color_msg,
                depth_msg=depth_msg,
                camera_info_msg=camera_info_msg,
                image_height=self.image_height,
                image_width=self.image_width,
                ws_margin_lr=self.mask_margin_lr,
                ws_margin_tb=self.mask_margin_tb,
                output_dir=os.path.join(self.output_dir, str(identifier)),
                identifier=str(identifier),
                enable_color=self.enable_color,
                enable_depth=self.enable_depth,
                enable_camera_info=self.enable_camera_info,
                use_mask=self.use_mask
            )
            rospy.loginfo(f"Frame {identifier}: Got input from Intel Realsense D455")
            
            box_filter, pred_label = get_groundingdino_inference(
                config_path=self.groundingdino_config_path,
                checkpoint_path=self.groundingdino_checkpoint_path,
                text_prompt=self.groundingdino_text_prompt,
                box_threshold=self.groundingdino_box_threshold,
                text_threshold=self.groundingdino_text_threshold,
                output_dir=os.path.join(self.output_dir, str(identifier)),
                color_image=color_image_np,
                token_spans=self.groundingdino_token_spans,
                cpu_only=self.groundingdino_cpu_only
            )

            # # If using GPU, ensure the data is on the correct device
            # if self.use_gpu:
            #     # Note: This step depends on how your get_groundingdino_inference function works
            #     # You might need to modify that function to accept a device parameter
            #     box_filter, pred_label = get_groundingdino_inference(
            #         config_path=self.config_path,
            #         checkpoint_path=self.checkpoint_path,
            #         text_prompt=self.text_prompt,
            #         box_threshold=self.box_threshold,
            #         text_threshold=self.text_threshold,
            #         output_dir=os.path.join(self.output_dir, str(identifier)),
            #         color_image=color_image_np,
            #         token_spans=self.token_spans,
            #         cpu_only=self.cpu_only,
            #         device=self.device  # Pass device to inference function
            #     )
            # else:
            #     # Original call for CPU
            #     box_filter, pred_label = get_groundingdino_inference(
            #         config_path=self.config_path,
            #         checkpoint_path=self.checkpoint_path,
            #         text_prompt=self.text_prompt,
            #         box_threshold=self.box_threshold,
            #         text_threshold=self.text_threshold,
            #         output_dir=os.path.join(self.output_dir, str(identifier)),
            #         color_image=color_image_np,
            #         token_spans=self.token_spans,
            #         cpu_only=self.cpu_only
            #     )
                
            rospy.loginfo(f"Frame {identifier}: Completed groundingdino inference")

            get_graspnet_inference(
                checkpoint_path=self.graspnet_checkpoint_path,
                refine_approach_dist=self.grapnet_refine_approach_dist,
                dist_thresh=self.graspnet_dist_thresh,
                angle_thresh=self.graspnet_angle_thresh,
                mask_thresh=self.graspnet_mask_thresh,
                color_image=color_image_np,
                depth_image=depth_image_np,
                camera_info=camera_info,
                box_filter=box_filter[0]    # Pass the first index of the box filter since it only accept Tensor(0,4) not Tensor(1,4)
            )
            rospy.loginfo(f"Frame {identifier}: Completed graspnet inference")

            # Clear CUDA cache periodically to avoid memory issues
            if self.use_gpu and identifier % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            rospy.logerr(f"Failed to process frame {identifier}: {e}")
            if self.use_gpu:
                torch.cuda.empty_cache()  # Try to clear GPU memory on error


if __name__ == '__main__':
    try:
        OVGNetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass