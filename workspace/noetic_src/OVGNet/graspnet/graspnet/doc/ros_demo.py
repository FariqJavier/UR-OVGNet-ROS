""" Demo to show prediction results.
    Author: chenxi-wang
"""

import rospy
import cv2
import numpy as np
import torch
import open3d as o3d
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from graspnetAPI import GraspGroup
from models.graspnet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo as GraspCameraInfo, create_point_cloud_from_depth_image
from PIL import Image as PILImage
import os
import message_filters


class GraspNetNode:
    def __init__(self):
        rospy.init_node('graspnet_node')

        # Parameters
        self.checkpoint_path = rospy.get_param('~checkpoint_path')
        self.num_point = rospy.get_param('~num_point', 20000)
        self.num_view = rospy.get_param('~num_view', 300)
        self.collision_thresh = rospy.get_param('~collision_thresh', 0.01)
        self.voxel_size = rospy.get_param('~voxel_size', 0.01)
        self.data_dir = rospy.get_param('~data_dir', 'doc/example_data')
        self.factor_depth = rospy.get_param('~factor_depth', 1000.0)

        # Load workspace mask once
        mask_path = os.path.join(self.data_dir, 'workspace_mask.png')
        self.workspace_mask = np.array(PILImage.open(mask_path)) if os.path.exists(mask_path) else None

        # Initialize model
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
        rospy.loginfo(f"Loaded checkpoint {self.checkpoint_path} (epoch {checkpoint['epoch']})")

        # ROS setup
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.camera_info = None

        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
        sync.registerCallback(self.synced_image_callback)

        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def color_callback(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.try_process()
        except Exception as e:
            rospy.logerr(f"Color callback error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg)
            self.try_process()
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")

    def synced_image_callback(self, color_msg, depth_msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg)
            if self.camera_info:
                self.process_frames()
        except Exception as e:
            rospy.logerr(f"Synced callback error: {e}")

    def try_process(self):
        if self.latest_color is not None and self.latest_depth is not None and self.camera_info is not None:
            self.process_frames()

    def process_frames(self):
        try:
            color = self.latest_color.astype(np.float32) / 255.0
            depth = self.latest_depth.astype(np.float32)

            K = np.array(self.camera_info.K).reshape(3, 3)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            camera = GraspCameraInfo(
                width=self.camera_info.width,
                height=self.camera_info.height,
                fx=fx, fy=fy, cx=cx, cy=cy, factor_depth=self.factor_depth
            )

            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

            mask = (depth > 0)
            if self.workspace_mask is not None:
                mask &= self.workspace_mask

            cloud_masked = cloud[mask]
            color_masked = color[mask]

            if len(cloud_masked) >= self.num_point:
                idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)

            cloud_sampled = torch.from_numpy(cloud_masked[idxs][np.newaxis].astype(np.float32)).to(self.device)
            end_points = {
                'point_clouds': cloud_sampled,
                'cloud_colors': color_masked[idxs]
            }

            gg = self.get_grasps(end_points)
            if self.collision_thresh > 0:
                gg = self.collision_detection(gg, cloud_masked)

            gg.nms()
            gg.sort_by_score()
            gg = gg[:50]

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
            pc.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([pc, *grippers])

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")

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
        GraspNetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
