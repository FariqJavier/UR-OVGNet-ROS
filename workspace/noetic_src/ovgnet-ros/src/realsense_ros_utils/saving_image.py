import rospy
import sensor_msgs
from cv_bridge import CvBridge
import cv2
import scipy.io as scio
import os
import numpy as np
from typing import Union
import tf2_ros
from tf.transformations import quaternion_multiply, quaternion_from_euler
from scipy.spatial.transform import Rotation as R

def create_and_publish_mask(
        height: int = 450,
        width: int = 848,
        margin_lr: float = 0.1,  # 10% margin from sides
        margin_tb: float = 0.1,  # 10% margin from top/bottom
        output_dir: str = '/tmp/ros_images',
        identifier: str =  '1'
    ) -> np.ndarray:
    """
    Create a workspace mask with a rectangular ROI
    Args:
        height: Height of mask (default 270)
        width: Width of mask (default 480)
        margin_lr: Left/right margin as a fraction of width (default 0.1)
        margin_tb: Top/bottom margin as a fraction of height (default 0.1)
        output_dir: Directory to save the mask image (default '/tmp/ros_images/')
        identifier: Unique identifier for the mask image (default 'workspace_mask')
    return:
        np.ndarray workspace mask
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

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
        full_mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Save resized mask (optional)
        cv2.imwrite(os.path.join(output_dir, f'workspace_mask_{identifier}.png'), full_mask)
        rospy.loginfo(f"Workspace mask saved to {os.path.join(output_dir, f'workspace_mask_{identifier}.png')}")

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
        workspace_mask (np.ndarray): Workspace mask template
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

        # # Create a unique subdirectory for this image
        # image_dir = os.path.join(output_dir, str(identifier))
        # os.makedirs(image_dir, exist_ok=True)

        # Save images
        color_path = os.path.join(output_dir, f'raw_color_{identifier}.png')
        depth_path = os.path.join(output_dir, f'raw_depth_{identifier}.png')

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

        np.save(os.path.join(output_dir, f'raw_depth_{identifier}.npy'), depth_np)  # Save as numpy array to preserve float values
        cv2.imwrite(depth_path, (depth_np * 1000).astype(np.uint16))  # Save visualization as PNG

        meta = {
            'intrinsic_matrix': np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ], dtype=np.float32),
            'image_size': color_image.shape[:2]
        }
    
        scio.savemat(os.path.join(output_dir, f'meta_{identifier}.mat'), meta)  # Save metadata as .mat file

        # Load workspace mask
        workspace_mask = workspace_mask > 0  # Convert to boolean mask

        # Apply mask to original color image for visualization
        masked_color_image = np.zeros_like(color_image)
        masked_color_image[workspace_mask] = color_image[workspace_mask]

        # Apply mask to original depth image for visualization
        masked_depth_image = np.zeros_like(depth_image)
        masked_depth_image[workspace_mask] = depth_image[workspace_mask]

        # Save masked image
        cv2.imwrite(os.path.join(output_dir, f'masked_color_{identifier}.png'), cv2.cvtColor(masked_color_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'masked_depth_{identifier}.png'), (masked_depth_image * 1000).astype(np.uint16))
        np.save(os.path.join(output_dir, f'masked_depth_{identifier}.npy'), masked_depth_image)

        rospy.loginfo(f"Saved images and metadata to {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to save images: {e}")

def load_color_and_depth_image(
        directory: str,
        identifier: str,
        enable_color: bool = True,
        enable_depth: bool = False,
        enable_camera_info: bool = False,
        use_mask: bool = True,
        use_numpy_depth: bool = True
    ) -> tuple[Union[str, None], Union[str, None], Union[str, None]]:
    """
    Load color and depth images from specified paths.
    Args:
        directory (str): Directory containing the image 
        identifier (str): Unique identifier for the image set.
        enable_color (bool): Flag to load color image.
        enable_depth (bool): Flag to load depth image.
        enable_camera_info (bool): Flag to load camera intrinsic parameters.
        use_mask (bool): Flag to load masked images.
    Returns:
        Tuple: Paths to the color image, depth image, and camera info path.
    """
    try:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Input directory '{directory}' not found.")
        if not identifier:
            raise ValueError("Identifier must be provided.")
        
        color_image_path = get_color_image(directory, identifier, use_mask) if enable_color else None
        depth_image_path = get_depth_image(directory, identifier, use_mask, use_numpy_depth) if enable_depth else None
        camera_info_path = get_camera_info(directory, identifier) if enable_camera_info else None

        return color_image_path, depth_image_path, camera_info_path

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
        color_image_path = os.path.join(directory, f'{prefix}_color_{identifier}.png')
        return color_image_path
    except Exception as e:
        raise RuntimeError(f"Failed to find masked color image: {e}")

def get_depth_image(
    directory: str, 
    identifier: str,
    use_mask: str =True,
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
        depth_image_path = os.path.join(directory, f'{prefix}_depth_{identifier}.png') if not use_numpy else os.path.join(directory, f'{prefix}_depth_{identifier: d}.npy')
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
        return os.path.join(directory, f'meta_{identifier}.mat')
    except Exception as e:
        raise RuntimeError(f"Failed to find camera info file: {e}")
        
def get_color_and_depth_image(
    color_msg: sensor_msgs.msg.Image, 
    depth_msg: sensor_msgs.msg.Image, 
    camera_info_msg: sensor_msgs.msg.CameraInfo, 
    output_dir: str,
    identifier: str,
    workspace_mask: np.ndarray = None,
    enable_color: bool = True,
    enable_depth: bool = False,
    enable_camera_info: bool = False,
    use_mask: bool = True,
) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[any, None]]:
    """
    Fixed version that properly handles camera transforms for eye-in-hand setup
    """
    
    # Setup TF listener
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    try:
        if not color_msg or not depth_msg or not camera_info_msg:
            raise ValueError("Received empty image messages")
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Output directory '{output_dir}' not found.")
        if not identifier:
            raise ValueError("Identifier must be provided.")
        
        bridge = CvBridge()
        
        # Convert ROS Image messages to OpenCV images
        raw_color_image = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        raw_depth_image = bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
        
        # Save images
        raw_color_path = os.path.join(output_dir, f'raw_color_{identifier}.png')
        raw_depth_path = os.path.join(output_dir, f'raw_depth_{identifier}.png')
        cv2.imwrite(raw_color_path, raw_color_image)
        
        # Get camera intrinsic parameters
        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        cx = camera_info_msg.K[2]
        cy = camera_info_msg.K[5]
        intrinsic_matrix = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)
        
        # FIX 1: Get the correct transform direction
        # For point cloud transformation, we need camera_link to base_link
        try:
            # Wait for transform to be available
            rospy.loginfo("Waiting for camera transform...")
            transform = tfBuffer.lookup_transform('base_link', 'camera_color_optical_frame', rospy.Time(0), rospy.Duration(5.0))
            
            # Debug the transform
            rospy.loginfo(f"Camera frame: {transform.child_frame_id}")
            rospy.loginfo(f"Base frame: {transform.header.frame_id}")
            
            # Extract position and orientation
            position_camera = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # FIX 2: Store quaternion in consistent format [x, y, z, w]
            orientation_camera = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            # FIX 3: Debug the camera orientation
            r = R.from_quat(orientation_camera)
            rotation_matrix = r.as_matrix()
            
            rospy.loginfo(f"Camera position: {position_camera}")
            rospy.loginfo(f"Camera rotation matrix:\n{rotation_matrix}")
            
            # Check if camera is pointing downward (for eye-in-hand)
            z_axis = rotation_matrix[:, 2]  # Camera's Z-axis
            rospy.loginfo(f"Camera Z-axis (should point down): {z_axis}")
            
            if z_axis[2] > -0.5:
                rospy.logwarn("WARNING: Camera might not be pointing downward properly!")
                rospy.logwarn(f"Z-component: {z_axis[2]} (should be negative)")
            
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup failed: {e}")
            rospy.logerr("Available frames:")
            rospy.logerr(tfBuffer.all_frames_as_string())
            raise
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform extrapolation failed: {e}")
            raise
        
        # Process depth image
        if raw_depth_image.dtype == np.uint16:
            # Depth in millimeters (RealSense/ZED/others)
            depth_np = raw_depth_image.astype(np.float32) / 1000.0 
        elif raw_depth_image.dtype == np.float32:
            # Depth already in meters
            depth_np = raw_depth_image.copy()
        else:
            raise ValueError(f"Unsupported depth image type: {raw_depth_image.dtype}")
        
        # Save depth data
        np.save(os.path.join(output_dir, f'raw_depth_{identifier}.npy'), depth_np)
        cv2.imwrite(raw_depth_path, (depth_np * 1000).astype(np.uint16))
        
        # FIX 4: Improved workspace limits for eye-in-hand
        # These should be in the camera frame, not global frame
        workspace_limits = np.asarray([
            [-0.5, 0.5],    # X-axis limits (left-right from camera view)
            [-0.4, 0.4],    # Y-axis limits (up-down from camera view)  
            [0.2, 1.0]      # Z-axis limits (depth from camera)
        ])
        
        # Create metadata with proper formatting
        meta = {
            'intrinsic_matrix': intrinsic_matrix,
            'position': position_camera,
            'orientation': orientation_camera,  # [x, y, z, w] format
            'workspace_limits': workspace_limits,
            'pixel_size': 1 / fx,
            'image_size': raw_color_image.shape[:2],
            # Add debugging info
            'transform_info': {
                'source_frame': transform.header.frame_id,
                'target_frame': transform.child_frame_id,
                'timestamp': transform.header.stamp.to_sec()
            }
        }
        
        # Save metadata
        scio.savemat(os.path.join(output_dir, f'meta_{identifier}.mat'), meta)
        
        # Process workspace mask if provided
        if workspace_mask is not None:
            workspace_mask = workspace_mask > 0
            
            # Apply mask to images
            masked_color_image = np.zeros_like(raw_color_image)
            masked_color_image[workspace_mask] = raw_color_image[workspace_mask]
            
            masked_depth_image = np.zeros_like(depth_np)
            masked_depth_image[workspace_mask] = depth_np[workspace_mask]
            
            # Save masked images
            cv2.imwrite(os.path.join(output_dir, f'masked_color_{identifier}.png'), 
                       cv2.cvtColor(masked_color_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, f'masked_depth_{identifier}.png'), 
                       (masked_depth_image * 1000).astype(np.uint16))
            np.save(os.path.join(output_dir, f'masked_depth_{identifier}.npy'), masked_depth_image)
        else:
            masked_color_image = raw_color_image
            masked_depth_image = depth_np
        
        rospy.loginfo(f"Saved images and metadata to {output_dir}")
        
        # Return appropriate data based on flags
        color_image = (masked_color_image if use_mask else raw_color_image) if enable_color else None
        depth_image = (masked_depth_image if use_mask else depth_np) if enable_depth else None
        camera_info = meta if enable_camera_info else None
        
        return color_image, depth_image, camera_info
        
    except Exception as e:
        rospy.logerr(f"Failed to get color and depth image: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        raise RuntimeError(f"Failed to get color and depth image: {e}")