#!/usr/bin/env python3
import sys
import os
import traceback

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, SRC_PATH)

import rospy
import cv2
import numpy as np
import torch
import open3d as o3d
import scipy.io as scio
from cv_bridge import CvBridge
from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from PIL import Image 

def normalize_depth(depth):
    depth = np.nan_to_num(depth)
    d_min, d_max = np.min(depth), np.max(depth)
    norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

class OneShotGraspNet:
    def __init__(self):
        rospy.init_node('graspnet_oneshot')

        self.checkpoint_path = rospy.get_param('~checkpoint_path')
        self.num_point = rospy.get_param('~num_point', 20000)
        self.num_view = rospy.get_param('~num_view', 300)
        self.collision_thresh = rospy.get_param('~collision_thresh', 0.01)
        self.voxel_size = rospy.get_param('~voxel_size', 0.01)
        self.data_dir = rospy.get_param('~data_dir', 'doc/example_data')
        self.factor_depth = rospy.get_param('~factor_depth', 1000.0)
        self.grasp_pose_path = os.path.join('~data-dir', 'demo_ros_result.png')

        self.bridge = CvBridge()

        # Load workspace mask
        mask_path = os.path.join(self.data_dir, 'workspace_mask.png')
        self.workspace_mask = np.array(Image.open(mask_path)) if os.path.exists(mask_path) else None

        # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = GraspNet(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()

        rospy.loginfo("Model loaded. Waiting for image and depth...")

        self.process()

    def process(self):
        # load data
        color = np.array(Image.open(os.path.join(self.data_dir, '2c.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(self.data_dir, '2d.png')))
        # depth = np.load(os.path.join(self.data_dir, 'depth_ros.npy'))
        # workspace_mask = np.array(Image.open(os.path.join(self.data_dir, 'workspace_mask.png')))
        workspace_mask = None
        meta = scio.loadmat(os.path.join(self.data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # generate cloud
        # camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        mask = (depth > 20) & (depth < 2000) 
        # min_depth = 0.2  # 20cm
        # max_depth = 1.5  # 1.5m
        # mask = (depth > min_depth) & (depth < max_depth)
        if workspace_mask is not None:
            mask &= workspace_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        if len(cloud_masked) < 10:
            rospy.logwarn("Too few valid points!")
            return

        # Sample point cloud
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, cloud_masked)

        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Grasp Pose", width=640, height=480)
        vis.add_geometry(cloud)
        for g in grippers:
            vis.add_geometry(g)
        vis.run()
        vis.destroy_window()

        # vis.poll_events()
        # vis.update_renderer()

        # # Capture image
        # vis.capture_screen_image(self.grasp_pose_path)

        # # Clean up
        # vis.destroy_window()

    def get_grasps(self, end_points):
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        return GraspGroup(grasp_preds[0].cpu().numpy())

    def collision_detection(self, gg, cloud):
        detector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        mask = detector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        return gg[~mask]


if __name__ == '__main__':
    try:
        OneShotGraspNet()
    except rospy.ROSInterruptException:
        pass
