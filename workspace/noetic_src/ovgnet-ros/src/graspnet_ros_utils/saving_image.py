import rospy
import sensor_msgs
from cv_bridge import CvBridge
import cv2
import scipy.io as scio
import os
import numpy as np

def create_and_publish_mask(
        height: int = 450,
        width: int = 848,
        margin_lr: float = 0.1,  # 10% margin from sides
        margin_tb: float = 0.1,  # 10% margin from top/bottom
        output_dir: str = '/tmp/ros_images',
        identifier: int = 1
    ):
    """
    Create a workspace mask with a rectangular ROI
    Args:
        height: Height of mask (default 270)
        width: Width of mask (default 480)
        margin_lr: Left/right margin as a fraction of width (default 0.1)
        margin_tb: Top/bottom margin as a fraction of height (default 0.1)
        output_dir: Directory to save the mask image (default '/tmp/ros_images/')
        identifier: Unique identifier for the mask image (default 'workspace_mask')
    """
    try:
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define workspace region (adjust these values based on your needs)
        margin_x = int(width * margin_lr)  # 10% margin from sides
        margin_y = int(height * margin_tb)  # 10% margin from top/bottom

        # Create a unique subdirectory for this image
        image_dir = os.path.join(output_dir, str(identifier))
        os.makedirs(image_dir, exist_ok=True)
        
        # Create rectangle ROI
        x1 = margin_x
        y1 = margin_y
        x2 = width - margin_x
        y2 = height - margin_y
        
        # Fill rectangle with white (255)
        full_mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Save resized mask (optional)
        cv2.imwrite(os.path.join(image_dir, f'workspace_mask_{identifier:d}.png'), full_mask)
        rospy.loginfo(f"Workspace mask saved to {os.path.join(image_dir, f'workspace_mask_{identifier:d}.png')}")

        return full_mask
    except Exception as e:
        raise RuntimeError(f"Failed to create and save workspace mask: {e}")

def save_color_and_depth_image(
        color_msg: sensor_msgs.msg.Image, 
        depth_msg: sensor_msgs.msg.Image, 
        camera_info_msg: sensor_msgs.msg.CameraInfo, 
        output_dir: str,
        identifier: str,
        workspace_mask: np.ndarray = None
    ):
    """ 
    Saves color and depth image to unique directories. 
    Args:
        color_msg (sensor_msgs.msg.Image): Color image message.
        depth_msg (sensor_msgs.msg.Image): Depth image message.
        camera_info_msg (sensor_msgs.msg.CameraInfo): Camera info message.
        output_dir (str): Directory to save images.
        identifier (str): Unique identifier for the image set.
    """
    try:
        if not color_msg or not depth_msg or not camera_info_msg or not identifier:
            raise ValueError("Received empty image messages")
        
        bridge = CvBridge()

        # Convert ROS Image messages to OpenCV images
        color_image = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)

        # Get camera intrinsic parameters
        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        cx = camera_info_msg.K[2]
        cy = camera_info_msg.K[5]

        # Create a unique subdirectory for this image
        image_dir = os.path.join(output_dir, str(identifier))
        os.makedirs(image_dir, exist_ok=True)

        # Save images
        color_path = os.path.join(image_dir, f'raw_color_{identifier:d}.png')
        depth_path = os.path.join(image_dir, f'raw_depth_{identifier:d}.png')

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

        np.save(os.path.join(image_dir, f'raw_depth_{identifier:d}.npy'), depth_np)  # Save as numpy array to preserve float values
        cv2.imwrite(depth_path, (depth_np * 1000).astype(np.uint16))  # Save visualization as PNG

        meta = {
            'intrinsic_matrix': np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ], dtype=np.float32),
            'image_size': color_image.shape[:2]
        }
    
        scio.savemat(os.path.join(image_dir, f'meta_{identifier:d}.mat'), meta)  # Save metadata as .mat file

        # Load workspace mask
        workspace_mask = workspace_mask > 0  # Convert to boolean mask

        # Apply mask to original color image for visualization
        masked_color_image = np.zeros_like(color_image)
        masked_color_image[workspace_mask] = color_image[workspace_mask]

        # Apply mask to original depth image for visualization
        masked_depth_image = np.zeros_like(depth_image)
        masked_depth_image[workspace_mask] = depth_image[workspace_mask]

        # Save masked image
        cv2.imwrite(os.path.join(image_dir, f'masked_color_{identifier:d}.png'), cv2.cvtColor(masked_color_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(image_dir, f'masked_depth_{identifier:d}.png'), (masked_depth_image * 1000).astype(np.uint16))
        np.save(os.path.join(image_dir, f'masked_depth_{identifier:d}.npy'), masked_depth_image)

        rospy.loginfo(f"Saved images and metadata to {image_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to save images: {e}")
