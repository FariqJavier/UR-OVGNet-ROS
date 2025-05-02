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

def load_color_and_depth_image(
        input_dir: str,
        len_subdir: int,
        identifier: str,
        enable_color: bool = True,
        enable_depth: bool = False,
        enable_camera_info: bool = False,
        use_mask: bool = True
    ):
    """
    Load color and depth images from specified paths.
    Args:
        input_dir (str): Directory containing the data image subdirectory set.
        len_subdir (int): Lenght of how many data image subdirectory set.
        identifier (str): Unique identifier for the image set.
        enable_color (bool): Flag to load color image.
        enable_depth (bool): Flag to load depth image.
        enable_camera_info (bool): Flag to load camera intrinsic parameters.
        use_mask (bool): Flag to load masked images.
    Returns:
        list
    """
    try:
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory '{input_dir}' not found.")
        if not len_subdir or len_subdir < 1:
            raise ValueError("Invalid number of subdirectories specified.")
        if not identifier:
            raise ValueError("Identifier must be provided.")

        color_images_paths = []
        depth_images_paths = []
        camera_info_paths = []
            
        # Process each subdirectory in order
        for subdir_num in range(1, len_subdir + 1):  # 1 to 13
            subdir_path = os.path.join(self.input_dir, str(subdir_num))

            if not os.path.isdir(subdir_path):
                raise FileNotFoundError(f"Subdirectory '{subdir_num}' not found in '{self.input_dir}'")

            rospy.loginfo(f"Processing subdirectory: {subdir_path}")
            
            color_image_path = get_color_image(subdir_path, identifier, use_mask) if enable_color else None
            depth_image_path = get_depth_image(subdir_path, identifier, use_mask) if enable_depth else None
            camera_info_path = get_camera_info(subdir_path, identifier) if enable_camera_info else None

            color_images_paths.append(color_image_path)
            depth_images_paths.append(depth_image_path)
            camera_info_paths.append(camera_info_path)

        return color_images_paths, depth_images_paths, camera_info_paths

    except Exception as e:
        raise RuntimeError(f"Failed to load images: {e}")

def get_color_image(
    directory: str, 
    identifier: str,
    use_mask: str =True
    ):
    """
    Get the color image path in the specified directory.
    Args:
        directory (str): Directory containing the data image subdirectory set.
        identifier (str): Unique identifier for the image set.
        use_mask (bool): Flag to load masked images.
    Returns:
        str: Path to the color image.
    """
    try:
        prefix = 'masked' if use_mask else 'raw'
        return color_image_path = os.path.join(directory, f'{prefix}_color_{identifier: d}.png')
    except Exception as e:
        raise RuntimeError(f"Failed to find masked color image: {e}")

def get_depth_image(
    directory: str, 
    identifier: str,
    use_mask: str =True
    use_numpy: str =True
    ):
    """
    Get the depth image path in the specified directory.
    Args:
        directory (str): Directory containing the data image subdirectory set.
        identifier (str): Unique identifier for the image set.
        use_mask (bool): Flag to load masked images.
    Returns:
        str: Path to the depth image.
    """
    try:
        prefix = 'masked' if use_mask else 'raw'
        depth_image_path = os.path.join(directory, f'{prefix}_depth_{identifier: d}.png') if not use_numpy else os.path.join(directory, f'{prefix}_depth_{identifier: d}.npy')
        return depth_image_path
    except Exception as e:
        raise RuntimeError(f"Failed to find masked depth image: {e}")

def get_camera_info(
    directory: str, 
    identifier: str
    ):
    """
    Get the camera info path in the specified directory.
    Args:
        directory (str): Directory containing the data image subdirectory set.
        identifier (str): Unique identifier for the image set.
    Returns:
        str: Path to the camera info file.
    """
    try:
        return os.path.join(directory, f'meta_{identifier: d}.mat')
    except Exception as e:
        raise RuntimeError(f"Failed to find camera info file: {e}")