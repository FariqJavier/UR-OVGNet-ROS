""" Demo to show prediction results.
    Author: chenxi-wang
"""

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image

DATA_DIR = 'doc/example_data'

class GraspNetNode:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
        parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
        parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
        parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
        parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
        self.cfgs = parser.parse_args()

        rospy.init_node('graspnet_node')

        # Init the model
        self.net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.cfgs.checkpoint_path, start_epoch))
        # set model to eval mode
        self.net.eval()

        self.bridge = CvBridge()
        
        # Subscribers
        self.color_sub = rospy.Subscriber(
            '/camera/color/image_raw', 
            Image, 
            self.color_callback,
            queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_rect_raw', 
            Image, 
            self.depth_callback,
            queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            '/camera/color/camera_info',
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )

        # Class variables
        self.latest_color = None
        self.latest_depth = None
        self.camera_info = None

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frames()
        except Exception as e:
            rospy.logerr(f"Color callback error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg)
            self.process_frames()
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")

    def get_processed_frames(self):
        if self.latest_color is None or self.latest_depth is None or self.camera_info is None:
            return
        
        while not rospy.is_shutdown():
            self.process_frames()
            rospy.sleep(0.1)

    def process_frames(self):
        try:
            # Prepare RGB and depth
            color = self.latest_color.astype(np.float32) / 255.0
            depth = self.latest_depth.astype(np.float32)

            # OPTIONAL: Workspace Mask
            workspace_mask = np.array(Image.open(os.path.join(DATA_DIR, 'workspace_mask.png')))

            # Intrinsics
            K = np.array(self.camera_info.K).reshape(3, 3)
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            factor_depth = 1000.0  # Or 1.0 depending on camera

            camera = CameraInfo(
                width=self.camera_info.width,
                height=self.camera_info.height,
                fx=fx, fy=fy, cx=cx, cy=cy, factor_depth=factor_depth
            )

            # Point cloud generation
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

            # Masking (optional if you don't have workspace_mask anymore)
            mask = (workspace_mask & (depth > 0))
            cloud_masked = cloud[mask]
            color_masked = color[mask]

            # sample points
            if len(cloud_masked) >= self.cfgs.num_point:
                idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point-len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]
            color_sampled = color_masked[idxs]

            # convert data
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
            cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
            end_points = dict()
            cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
            cloud_sampled = cloud_sampled.to(self.device)
            end_points['point_clouds'] = cloud_sampled
            end_points['cloud_colors'] = color_sampled

            gg = self.get_grasps(end_points)
            if self.cfgs.collision_thresh > 0:
                gg = self.collision_detection(gg, np.array(cloud.points))
            gg.nms()
            gg.sort_by_score()
            gg = gg[:50]
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
    
    def get_grasps(self, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
    
    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
        gg = gg[~collision_mask]
        return gg

if __name__ == '__main__':
    try:
        graspNetNode = GraspNetNode()
        graspNetNode.get_processed_frames()
    except rospy.ROSInterruptException:
        pass
