import math
import numpy as np
import pybullet as p
import cv2
import open3d as o3d
import open3d_plus as o3dp
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# from constants import WORKSPACE_LIMITS
from scipy.spatial.distance import cdist
import random
import rospy
import copy
import tf2_ros

reconstruction_config = {
    'nb_neighbors': 60,        # Increased from 50
    'std_ratio': 1.5,          # Decreased from 2.0
    'voxel_size': 0.005,       # Keep as is
    'icp_max_try': 5,          # Keep as is
    'icp_max_iter': 2000,      # Keep as is
    'translation_thresh': 3.95, # Keep as is
    'rotation_thresh': 0.02,    # Keep as is
    'max_correspondence_distance': 0.015, # Decreased from 0.02
    'use_poisson': False,      # Consider changing to True for smoother results
    'dbscan_eps': 0.02,       # Add this parameter (smaller for tighter clustering)
    'dbscan_min_points': 10,   # Add this parameter (higher for more strict clustering)
    'orient_for_grasping': True,
    'debug_orientation': False,
    # New parameters for table alignment
    'plane_distance_threshold': 0.01,   # Distance threshold for plane detection
    'visualize_alignment': False,       # Set to True for debugging
    'visualize_final': True,           # Set to True for debugging
    'min_object_height': 0.005,         # Min height above table
    'max_object_height': 0.5,           # Max height above table
    'final_voxel_size': 0.002           # Final voxel size for downsampling
}

graspnet_config = {
    'graspnet_checkpoint_path': 'graspnet/graspnet/logs/log_rs/checkpoint.tar',
    'refine_approach_dist': 0.01,
    'dist_thresh': 0.05,
    'angle_thresh': 15,
    'mask_thresh': 0.5
}

def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

# Helper function to create rotation matrices
def get_rotation_matrix(rx, ry, rz):
    """
    Create a rotation matrix from Euler angles (in degrees)
    
    Args:
        rx, ry, rz: Rotation angles in degrees around x, y, z axes
    Returns:
        4x4 transformation matrix
    """
    # Convert to radians
    rx, ry, rz = np.radians([rx, ry, rz])
    
    # Rotation matrices around each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Create 4x4 transform
    transform = np.eye(4)
    transform[:3, :3] = R
    
    return transform

def process_pcds(pcds, reconstruction_config):
    """
    Advanced point cloud fusion with global registration and fine-tuning.
    
    Args:
        pcds: List of point clouds to fuse
        reconstruction_config: Configuration parameters
    Returns:
        trans: Dictionary of transformations
        fused_pcd: Fused point cloud
    """
    if len(pcds) <= 1:
        return {0: np.eye(4)}, pcds[0] if pcds else None

    # Clean and prepare all point clouds
    processed_pcds = []
    for i, pcd in enumerate(pcds):
        # Make a copy to avoid modifying the original
        pcd_copy = copy.deepcopy(pcd)
        
        # Apply DBSCAN clustering to get the main object and remove background noise
        if len(pcd_copy.points) > 100:
            labels = np.array(pcd_copy.cluster_dbscan(
                eps=reconstruction_config.get('dbscan_eps', 0.02),
                min_points=reconstruction_config.get('dbscan_min_points', 10)
            ))
            
            if len(labels) > 0 and max(labels) >= 0:
                # Keep only the largest cluster
                largest_cluster = np.bincount(labels[labels >= 0]).argmax()
                pcd_copy = pcd_copy.select_by_index(np.where(labels == largest_cluster)[0])
        
        # Ensure the point cloud has normals
        pcd_copy.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=reconstruction_config.get('normal_radius', 0.02),
                max_nn=reconstruction_config.get('normal_max_nn', 30)
            )
        )
        
        # Apply voxel downsampling
        if reconstruction_config.get('voxel_size') and len(pcd_copy.points) > 0:
            pcd_copy = pcd_copy.voxel_down_sample(reconstruction_config['voxel_size'])
        
        if len(pcd_copy.points) > 50:
            processed_pcds.append(pcd_copy)
    
    if len(processed_pcds) <= 1:
        return {0: np.eye(4)}, processed_pcds[0] if processed_pcds else None
    
    # Find the point cloud with the most points to use as reference
    point_counts = [len(pcd.points) for pcd in processed_pcds]
    ref_idx = np.argmax(point_counts)
    target_pcd = processed_pcds[ref_idx]
    
    # Final fused point cloud
    fused_pcd = copy.deepcopy(target_pcd)
    transformations = {ref_idx: np.eye(4)}
    
    voxel_size = reconstruction_config.get('voxel_size', 0.005)
    
    # Compute FPFH features for target
    target_down = target_pcd.voxel_down_sample(voxel_size * 3)
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=100)
    )
    
    # Process each source point cloud
    for i, source_pcd in enumerate(processed_pcds):
        if i == ref_idx:
            continue
            
        print(f"Aligning point cloud {i} to reference...")
        
        # Compute FPFH features for source
        source_down = source_pcd.voxel_down_sample(voxel_size * 3)
        source_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=100)
        )
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=100)
        )
        
        # Global registration using RANSAC with FPFH features
        print("RANSAC global registration...")
        registration_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 10,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 10)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
        )
        
        print(f"Global registration fitness: {registration_result.fitness}")
        initial_transform = registration_result.transformation
        
        # Try multiple strategies if global registration fails
        if registration_result.fitness < 0.6:
            print("Global registration unsuccessful, trying alternative strategies...")
            
            # Try alternative rotations (90-degree intervals around each axis)
            rotations = []
            for rx in [0, 90, 180, 270]:
                for ry in [0, 90, 180, 270]:
                    for rz in [0]:  # Limiting Z rotation for efficiency
                        rotations.append(get_rotation_matrix(rx, ry, rz))
            
            best_fitness = 0
            best_transform = np.eye(4)
            
            for rotation in rotations:
                # Apply the rotation
                source_rotated = copy.deepcopy(source_down)
                source_rotated.transform(rotation)
                
                # Try ICP with this rotation
                icp_result = o3d.pipelines.registration.registration_icp(
                    source_rotated, target_down, voxel_size * 10, 
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                )
                
                if icp_result.fitness > best_fitness:
                    best_fitness = icp_result.fitness
                    best_transform = rotation @ icp_result.transformation
            
            if best_fitness > registration_result.fitness:
                initial_transform = best_transform
                print(f"Alternative strategy successful, fitness: {best_fitness}")
            
            # If still poor alignment, try point cloud subsampling
            if best_fitness < 0.3:
                print("Trying subsampling strategy...")
                
                # Take different subsets of points to find better alignment
                n_attempts = 10
                subset_ratio = 0.7
                
                for attempt in range(n_attempts):
                    # Randomly sample points
                    n_points = len(source_down.points)
                    indices = np.random.choice(n_points, int(n_points * subset_ratio), replace=False)
                    source_subset = source_down.select_by_index(indices)
                    
                    # Try ICP with this subset
                    icp_result = o3d.pipelines.registration.registration_icp(
                        source_subset, target_down, voxel_size * 10, 
                        np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                    )
                    
                    if icp_result.fitness > best_fitness:
                        best_fitness = icp_result.fitness
                        best_transform = icp_result.transformation
                
                if best_fitness > 0.3:
                    initial_transform = best_transform
                    print(f"Subsampling strategy successful, fitness: {best_fitness}")
        
        # Replace the fine ICP registration block with this improved version
        print("Performing fine ICP registration...")
        # Use a much larger correspondence distance at first to ensure matches
        initial_distance = voxel_size * 10  # Try a larger initial distance

        # Try multi-stage ICP with gradually decreasing distance thresholds
        current_transform = initial_transform
        for distance_multiplier in [10, 5, 2]:
            current_distance = voxel_size * distance_multiplier
            icp_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, current_distance,
                current_transform,  # Use the progressively refined transformation
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # First use point-to-point (more robust)
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50
                )
            )
            
            # Update the transformation if ICP improved the alignment
            if icp_result.fitness > 0.0:
                current_transform = icp_result.transformation
                print(f"ICP stage with distance {current_distance} achieved fitness: {icp_result.fitness}")
            else:
                print(f"ICP stage with distance {current_distance} failed")

        # Final fine ICP with point-to-plane for precision
        if current_transform is not initial_transform:  # Only if earlier stages worked
            fine_icp_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, voxel_size * 2,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=reconstruction_config.get('icp_max_iter', 100),
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )
            
            print(f"Final ICP fitness: {fine_icp_result.fitness}")
            final_transform = fine_icp_result.transformation
        else:
            # Fallback if ICP completely fails
            print("All ICP stages failed, using global registration result")
            final_transform = initial_transform
        
        # Update the transformation dictionary
        transformations[i] = final_transform
        
        # Transform and add to the fused point cloud
        transformed_source = copy.deepcopy(source_pcd)
        transformed_source.transform(final_transform)
        fused_pcd += transformed_source
    
    # Final cleanup of the fused point cloud
    print("Final cleanup and optimization...")

    # Remove outliers from the final point cloud
    if len(fused_pcd.points) > 200:
        fused_pcd, _ = fused_pcd.remove_statistical_outlier(
            nb_neighbors=reconstruction_config.get('nb_neighbors', 50),
            std_ratio=reconstruction_config.get('std_ratio', 2.0)
        )
         # Add radius outlier removal as well
        fused_pcd, _ = fused_pcd.remove_radius_outlier(
            nb_points=50,  # Require at least 50 points in neighborhood
            radius=0.02    # Within 2cm radius
        )
    
    # Apply Poisson surface reconstruction to get a smoother result
    if reconstruction_config.get('use_poisson', False) and len(fused_pcd.points) > 200:
        print("Applying Poisson surface reconstruction...")
        fused_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2,
                max_nn=30
            )
        )
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                fused_pcd, depth=8, width=0, scale=1.1, linear_fit=False
            )
        
        # Convert mesh back to point cloud
        pcd_from_mesh = mesh.sample_points_poisson_disk(
            number_of_points=len(fused_pcd.points),
            init_factor=5
        )
        fused_pcd = pcd_from_mesh
    
    # Final voxel downsampling to unify point density
    if reconstruction_config.get('voxel_size') and len(fused_pcd.points) > 30000:
        fused_pcd = fused_pcd.voxel_down_sample(reconstruction_config['voxel_size'])
    
    # Ensure normals for the final result
    fused_pcd.estimate_normals()
    
    print(f"Fusion complete. Final point cloud has {len(fused_pcd.points)} points.")
    return transformations, fused_pcd

def process_single_pcd(pcd, reconstruction_config):
    # Step 1: Apply statistical outlier removal to filter noise
    voxel_size = reconstruction_config['voxel_size']
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=reconstruction_config['nb_neighbors'],
        std_ratio=reconstruction_config['std_ratio']
    )

    # Step 2: Estimate normals
    pcd.estimate_normals()

    # Step 3: Apply voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size)

    # Step 4: Perform ICP registration (even if it's the same point cloud)
    transok_flag = False
    for _ in range(reconstruction_config['icp_max_try']):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd,
            pcd,  # Register the point cloud with itself
            reconstruction_config['max_correspondence_distance'],
            np.eye(4, dtype=np.float32),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
        )
        
        # If the registration is successful based on the thresholds
        if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) and \
           (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
            transok_flag = True
            break

    # If no good transformation found, reset to identity matrix
    if not transok_flag:
        reg_p2p.transformation = np.eye(4, dtype=np.float32)

    # Step 5: Apply the transformation (even if it's the identity matrix)
    pcd = pcd.transform(reg_p2p.transformation)

    # Return the transformation matrix and the processed point cloud
    return reg_p2p.transformation, pcd

# def get_fuse_pointcloud(env, box, color1, depth1):
#     # box[:2] -= box[2:] / 2
#     # box[2:] += box[:2]
#     # a = box.numpy()
#     # xtop = [a[0]*0.448-0.224, a[1]*0.448+0.276]
#     # xdown = [a[2]*0.448-0.224, a[3]*0.448+0.276]
#     # bounds = np.asarray([[xtop[1], xdown[1]], [xtop[0], xdown[0]], [-0.0001, 0.4]])
#     box[:2] -= box[2:] / 2
#     box[2:] += box[:2]
#     a = box.numpy()
#     xtop = [(a[0] * 0.448), (a[1] * 0.448)]
#     xdown = [(a[2] * 0.448), (a[3] * 0.448)]
#     # xtop = points[xtop[0], xtop[1]][:2]
#     # xdown = points[xdown[0], xdown[1]][:2]
#     bounds = np.asarray([[xtop[1] + 0.276, xdown[1] + 0.276], [xtop[0] - 0.224, xdown[0] - 0.224], [-0.0001, 0.4]])
#     # bounds = np.asarray([[xtop[1] + 0.25, xdown[1] + 0.302], [xtop[0] - 0.250, xdown[0] - 0.198], [-0.0001, 0.4]])
#     pcds = []
#     configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         color, depth, _ = env.render_camera(config)
#         xyz = get_pointcloud(depth, config["intrinsics"])
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = p.getMatrixFromQuaternion(config["rotation"])
#         rotation = np.array(rotation).reshape(3, 3)
#         transform = np.eye(4)
#         transform[:3, :] = np.hstack((rotation, position))
#         points = transform_pointcloud(xyz, transform)

#         # Filter out 3D points that are outside of the predefined bounds.
#         ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
#         iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
#         iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
#         valid = ix & iy & iz
#         points = points[valid]
#         colors = color[valid]
#         # Sort 3D points by z-value, which works with array assignment to simulate
#         # z-buffering for rendering the heightmap image.
#         iz = np.argsort(points[:, -1])
#         points, colors = points[iz], colors[iz]

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#         pcd.voxel_down_sample(reconstruction_config['voxel_size'])
#         # # visualization
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#         # the first pcd is the one for start fusion
#         pcds.append(pcd)

#     _, fuse_pcd = process_pcds(pcds, reconstruction_config)
#     # a = np.asarray(fuse_pcd.points)  # A已经变成n*3的矩阵
#     # ox = (a[Ellipsis, 0] >= bounds[0, 0]) & (a[Ellipsis, 0] < bounds[0, 1])
#     # oy = (a[Ellipsis, 1] >= bounds[1, 0]) & (a[Ellipsis, 1] < bounds[1, 1])
#     # oz = (a[Ellipsis, 2] >= bounds[2, 0]) & (a[Ellipsis, 2] < bounds[2, 1])
#     # alid = ox & oy & oz
#     # a = a[alid]
#     # cd = o3d.geometry.PointCloud()  # 实例化一个pointcloud类
#     # cd.points = o3d.utility.Vector3dVector(a)  # 给该类传入坐标数据，此时pcd.points已经是一个点云了
#     # o3d.visualization.draw_geometries([cd])  # 显示一下

#     # visualization
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([fuse_pcd, frame])


#     return fuse_pcd

def get_single_pointcloud(realsense_input, groundingdino_output, camera_id=None):
    box_filter = groundingdino_output.box_filter[0]
    rospy.loginfo(f"Camera {camera_id} - Box Filter: {box_filter}")

    # Convert box_filter to numpy if it's a tensor
    if hasattr(box_filter, 'numpy'):
            box_filter_np = box_filter.numpy() 
    else:
        box_filter_np = np.array(box_filter)
            
    # Extract box coordinates (center_x, center_y, width, height)
    center_x, center_y, width, height = box_filter_np
    
    # Get camera data
    color_image_np = realsense_input.color_image_np
    depth_image_np = realsense_input.depth_image_np
    camera_info = realsense_input.camera_info
    
    # Get image dimensions
    height_img, width_img = depth_image_np.shape[:2] if len(depth_image_np.shape) > 2 else depth_image_np.shape
    
    # Calculate pixel coordinates from normalized coordinates with margin
    margin_w = width * 0.1  # 10% margin
    margin_h = height * 0.1  # 10% margin
    
    x_min = max(0, (center_x - width/2 - margin_w) * width_img)
    y_min = max(0, (center_y - height/2 - margin_h) * height_img)
    x_max = min(width_img, (center_x + width/2 + margin_w) * width_img)
    y_max = min(height_img, (center_y + height/2 + margin_h) * height_img)
    
    # Convert to integers
    x_min, y_min = int(x_min), int(y_min)
    x_max, y_max = int(x_max), int(y_max)
    
    rospy.loginfo(f"Camera {camera_id} - Bounding box in pixels: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    
    # Create a mask for the pixels within the bounding box
    mask = np.zeros((height_img, width_img), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    
    # Apply mask to depth image
    masked_depth = np.copy(depth_image_np)
    masked_depth[~mask] = 0
    
    # Skip if no valid depth in the masked region
    if np.all(masked_depth == 0):
        rospy.logwarn(f"Camera {camera_id} - No valid depth data in bounding box region")
        pass
        
    # Convert masked depth to point cloud
    xyz = get_pointcloud(masked_depth, camera_info["intrinsic_matrix"])
    
    # Apply transform to world coordinates
    position = np.array(camera_info["position"]).reshape(3, 1)
    rotation = p.getMatrixFromQuaternion(camera_info["orientation"])
    rotation = np.array(rotation).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position.flatten()
    
    # Transform points to world coordinates
    transformed_points = transform_pointcloud(xyz, transform)
    
    # Filter out points with zero depth (outside the bounding box)
    valid_points = (transformed_points[:,:,2] > 0)
    if np.sum(valid_points) < 50:
        rospy.logwarn(f"Camera {camera_id} - Not enough valid points ({np.sum(valid_points)}) in bounding box")
        pass
        
    # Reshape for filtering
    points_shape = transformed_points.shape
    points_reshaped = transformed_points.reshape(-1, 3)
    mask_reshaped = mask.reshape(-1)
    
    # Filter points using the mask
    filtered_points = points_reshaped[mask_reshaped & (points_reshaped[:,2] > 0)]
    
    # Get corresponding colors
    color_reshaped = color_image_np.reshape(-1, 3)
    filtered_colors = color_reshaped[mask_reshaped & (points_reshaped[:,2] > 0)]
    
    # Log statistics
    if len(filtered_points) > 0:
        x_values = filtered_points[:, 0]
        y_values = filtered_points[:, 1]
        z_values = filtered_points[:, 2]
        
        rospy.loginfo(f"Camera {camera_id} - Filtered points stats:")
        rospy.loginfo(f"X: max={np.max(x_values)}, min={np.min(x_values)}, mean={np.mean(x_values)}")
        rospy.loginfo(f"Y: max={np.max(y_values)}, min={np.min(y_values)}, mean={np.mean(y_values)}")
        rospy.loginfo(f"Z: max={np.max(z_values)}, min={np.min(z_values)}, mean={np.mean(z_values)}")
        rospy.loginfo(f"Total valid points: {len(filtered_points)}")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)
    
    # Apply statistical outlier removal
    if len(filtered_points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        rospy.loginfo(f"Camera {camera_id} - After outlier removal: {len(pcd.points)} points")
    
    # Apply voxel downsampling if configured
    if len(pcd.points) > 0 and hasattr(pcd, 'voxel_down_sample') and reconstruction_config.get('voxel_size'):
        pcd = pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        rospy.loginfo(f"Camera {camera_id} - After downsampling: {len(pcd.points)} points")

    return pcd

def get_single_pointcloud_fixed(realsense_input, groundingdino_output, camera_id=None):
    """
    Fixed version of point cloud generation with proper error handling and debugging
    """
    try:
        box_filter = groundingdino_output.box_filter[0]
        rospy.loginfo(f"Camera {camera_id} - Box Filter: {box_filter}")

        # Convert box_filter to numpy if it's a tensor
        if hasattr(box_filter, 'numpy'):
                box_filter_np = box_filter.numpy() 
        else:
            box_filter_np = np.array(box_filter)
                
        # Extract box coordinates (center_x, center_y, width, height)
        center_x, center_y, width, height = box_filter_np
        
        # Debug the bounding box values
        rospy.loginfo(f"Camera {camera_id} - Detection: center=({center_x:.3f}, {center_y:.3f}), size=({width:.3f}, {height:.3f})")
        
        # Validate bounding box
        if width <= 0 or height <= 0:
            rospy.logerr(f"Camera {camera_id} - Invalid bounding box dimensions: width={width}, height={height}")
            return None
            
        if center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1:
            rospy.logerr(f"Camera {camera_id} - Invalid bounding box center: ({center_x}, {center_y})")
            return None
        
        # Get camera data
        color_image_np = realsense_input.color_image_np
        depth_image_np = realsense_input.depth_image_np
        camera_info = realsense_input.camera_info
        
        # Get image dimensions
        height_img, width_img = depth_image_np.shape[:2] if len(depth_image_np.shape) > 2 else depth_image_np.shape
        rospy.loginfo(f"Camera {camera_id} - Image dimensions: {width_img}x{height_img}")
        
        # FIX 1: Increase margin for better object capture
        margin_w = width * 0.08  # (8% margin)
        margin_h = height * 0.08  # (8% margin)
        
        # Calculate pixel coordinates from normalized coordinates with margin
        x_min = max(0, (center_x - width/2 - margin_w) * width_img)
        y_min = max(0, (center_y - height/2 - margin_h) * height_img)
        x_max = min(width_img, (center_x + width/2 + margin_w) * width_img)
        y_max = min(height_img, (center_y + height/2 + margin_h) * height_img)
        
        # Convert to integers
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)
        
        rospy.loginfo(f"Camera {camera_id} - Bounding box in pixels: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
        
        # FIX 2: Validate pixel coordinates
        if x_max <= x_min or y_max <= y_min:
            rospy.logerr(f"Camera {camera_id} - Invalid pixel bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            return None
        
        # Check depth image statistics before masking
        depth_valid = depth_image_np[depth_image_np > 0]
        if len(depth_valid) == 0:
            rospy.logerr(f"Camera {camera_id} - No valid depth data in entire image")
            return None
            
        rospy.loginfo(f"Camera {camera_id} - Depth range: {np.min(depth_valid):.3f} to {np.max(depth_valid):.3f}m")
        
        # Create a mask for the pixels within the bounding box
        mask = np.zeros((height_img, width_img), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        rospy.loginfo(f"Camera {camera_id} - Mask covers {np.sum(mask)} pixels ({100*np.sum(mask)/(width_img*height_img):.1f}% of image)")
        
        # Apply mask to depth image
        masked_depth = np.copy(depth_image_np)
        masked_depth[~mask] = 0
        
        # FIX 3: Check masked depth statistics
        masked_valid = masked_depth[masked_depth > 0]
        if len(masked_valid) == 0:
            rospy.logerr(f"Camera {camera_id} - No valid depth data in bounding box region")
            
            # Debug: Save depth image for inspection
            import cv2
            debug_depth = (depth_image_np / np.max(depth_image_np) * 255).astype(np.uint8)
            debug_depth_colored = cv2.applyColorMap(debug_depth, cv2.COLORMAP_JET)
            cv2.rectangle(debug_depth_colored, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imwrite(f'/tmp/debug_depth_cam_{camera_id}.png', debug_depth_colored)
            rospy.loginfo(f"Camera {camera_id} - Saved debug depth image to /tmp/debug_depth_cam_{camera_id}.png")
            
            return None
        
        rospy.loginfo(f"Camera {camera_id} - Masked depth range: {np.min(masked_valid):.3f} to {np.max(masked_valid):.3f}m, mean: {np.mean(masked_valid):.3f}m")
        rospy.loginfo(f"Camera {camera_id} - Valid masked pixels: {len(masked_valid)}")
            
        # Convert masked depth to point cloud
        xyz = get_pointcloud(masked_depth, camera_info["intrinsic_matrix"])
        
        # FIX 4: Debug camera frame points before transformation
        valid_camera_mask = xyz[:,:,2] > 0
        if np.sum(valid_camera_mask) == 0:
            rospy.logerr(f"Camera {camera_id} - No valid 3D points generated from depth")
            return None
            
        valid_camera_points = xyz[valid_camera_mask]
        rospy.loginfo(f"Camera {camera_id} - Camera frame points: {len(valid_camera_points)}")
        rospy.loginfo(f"Camera {camera_id} - Camera frame X: [{np.min(valid_camera_points[:,0]):.3f}, {np.max(valid_camera_points[:,0]):.3f}]")
        rospy.loginfo(f"Camera {camera_id} - Camera frame Y: [{np.min(valid_camera_points[:,1]):.3f}, {np.max(valid_camera_points[:,1]):.3f}]")
        rospy.loginfo(f"Camera {camera_id} - Camera frame Z: [{np.min(valid_camera_points[:,2]):.3f}, {np.max(valid_camera_points[:,2]):.3f}]")
        
        # Apply transform to world coordinates
        position = np.array(camera_info["position"]).reshape(3, 1)
        orientation = np.array(camera_info["orientation"])  # [x, y, z, w]
        
        # FIX 5: Use scipy for quaternion conversion instead of pybullet
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(orientation).as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position.flatten()
        
        rospy.loginfo(f"Camera {camera_id} - Transform:\nPosition: {position.flatten()}\nRotation matrix:\n{rotation}")
        
        # Transform points to world coordinates
        transformed_points = transform_pointcloud(xyz, transform)
        
        # FIX 6: Debug world frame points
        valid_world_mask = transformed_points[:,:,2] > -1.0  # Allow points slightly below table
        if np.sum(valid_world_mask) < 50:
            rospy.logwarn(f"Camera {camera_id} - Not enough valid points ({np.sum(valid_world_mask)}) after transformation")
            
            # Debug transformation
            rospy.loginfo(f"Camera {camera_id} - Camera position: {position.flatten()}")
            rospy.loginfo(f"Camera {camera_id} - First few transformed points:")
            flat_points = transformed_points.reshape(-1, 3)
            valid_flat = flat_points[flat_points[:,2] > -1.0]
            if len(valid_flat) > 0:
                for i in range(min(5, len(valid_flat))):
                    rospy.loginfo(f"  Point {i}: {valid_flat[i]}")
        
        # FIX 7: Improved point filtering
        # Reshape for filtering
        points_shape = transformed_points.shape
        points_reshaped = transformed_points.reshape(-1, 3)
        mask_reshaped = mask.reshape(-1)
        
        # Create combined mask: bounding box AND reasonable Z values
        reasonable_z_mask = (points_reshaped[:,2] > -0.3) & (points_reshaped[:,2] < 2.5)  # Between -30cm and 2.5m
        combined_mask = mask_reshaped & reasonable_z_mask
        
        # Filter points using the combined mask
        filtered_points = points_reshaped[combined_mask]
        
        # Get corresponding colors
        color_reshaped = color_image_np.reshape(-1, 3)
        filtered_colors = color_reshaped[combined_mask]
        
        # FIX 8: Check filtering results
        if len(filtered_points) == 0:
            rospy.logerr(f"Camera {camera_id} - No points after filtering")
            rospy.loginfo(f"Camera {camera_id} - Mask sum: {np.sum(mask_reshaped)}")
            rospy.loginfo(f"Camera {camera_id} - Reasonable Z sum: {np.sum(reasonable_z_mask)}")
            rospy.loginfo(f"Camera {camera_id} - Combined mask sum: {np.sum(combined_mask)}")
            return None
        
        # Log statistics
        x_values = filtered_points[:, 0]
        y_values = filtered_points[:, 1]
        z_values = filtered_points[:, 2]
        
        rospy.loginfo(f"Camera {camera_id} - Filtered points stats:")
        rospy.loginfo(f"X: max={np.max(x_values):.3f}, min={np.min(x_values):.3f}, mean={np.mean(x_values):.3f}")
        rospy.loginfo(f"Y: max={np.max(y_values):.3f}, min={np.min(y_values):.3f}, mean={np.mean(y_values):.3f}")
        rospy.loginfo(f"Z: max={np.max(z_values):.3f}, min={np.min(z_values):.3f}, mean={np.mean(z_values):.3f}")
        rospy.loginfo(f"Total valid points: {len(filtered_points)}")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)
        
        # FIX 9: More conservative outlier removal
        if len(filtered_points) > 100:
            original_count = len(pcd.points)
            pcd, outlier_indices = pcd.remove_statistical_outlier(reconstruction_config['nb_neighbors'], reconstruction_config['std_ratio'])  # Increased threshold
            rospy.loginfo(f"Camera {camera_id} - After outlier removal: {len(pcd.points)} points (removed {original_count - len(pcd.points)})")
        
        # FIX 10: Optional voxel downsampling (only if too many points)
        if len(pcd.points) > 50000:  # Only downsample if too many points
            # voxel_size = 0.002  # 2mm voxels
            original_count = len(pcd.points)
            pcd = pcd.voxel_down_sample(reconstruction_config['voxel_size'])
            rospy.loginfo(f"Camera {camera_id} - After downsampling: {len(pcd.points)} points (removed {original_count - len(pcd.points)})")
        
        # FIX 11: Ensure we still have a reasonable number of points
        if len(pcd.points) < 1000:
            rospy.logwarn(f"Camera {camera_id} - Too few points after processing: {len(pcd.points)}")
            return None
        
        rospy.loginfo(f"Camera {camera_id} - Successfully generated point cloud with {len(pcd.points)} points")
        
        # visualization
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([pcd, frame])
        return pcd
        
    except Exception as e:
        rospy.logerr(f"Camera {camera_id} - Error in point cloud generation: {str(e)}")
        import traceback
        rospy.logerr(traceback.format_exc())
        return None

def check_transform_pointcloud_function():
    """
    Test the transform_pointcloud function to ensure it works correctly
    """
    print("Testing transform_pointcloud function...")
    
    # Create test data
    test_xyz = np.random.rand(100, 100, 3) * 0.5  # Random points
    test_transform = np.eye(4)
    test_transform[:3, 3] = [1, 2, 3]  # Translation
    
    # Test transformation
    result = transform_pointcloud(test_xyz, test_transform)
    
    # Check if transformation was applied
    original_mean = np.mean(test_xyz.reshape(-1, 3), axis=0)
    result_mean = np.mean(result.reshape(-1, 3), axis=0)
    expected_mean = original_mean + [1, 2, 3]
    
    if np.allclose(result_mean, expected_mean, atol=0.001):
        print("✅ transform_pointcloud function is working correctly")
        return True
    else:
        print("❌ transform_pointcloud function has issues")
        print(f"Original mean: {original_mean}")
        print(f"Result mean: {result_mean}")
        print(f"Expected mean: {expected_mean}")
        return False

def visualize_detection_box_on_image(color_image, depth_image, box_filter, camera_id):
    """
    Visualize the detection box on the color and depth images for debugging
    """
    import cv2
    
    # Convert box to pixel coordinates
    height_img, width_img = depth_image.shape[:2]
    center_x, center_y, width, height = box_filter
    
    margin_w = width * 0.2
    margin_h = height * 0.2
    
    x_min = max(0, (center_x - width/2 - margin_w) * width_img)
    y_min = max(0, (center_y - height/2 - margin_h) * height_img)
    x_max = min(width_img, (center_x + width/2 + margin_w) * width_img)
    y_max = min(height_img, (center_y + height/2 + margin_h) * height_img)
    
    x_min, y_min = int(x_min), int(y_min)
    x_max, y_max = int(x_max), int(y_max)
    
    # Draw on color image
    color_vis = color_image.copy()
    cv2.rectangle(color_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(color_vis, f'Camera {camera_id}', (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw on depth image
    depth_vis = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.rectangle(depth_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite(f'/home/ros/data/detection_color_cam_{camera_id}.png', color_vis)
    cv2.imwrite(f'/home/ros/data/detection_depth_cam_{camera_id}.png', depth_vis)
    
    rospy.loginfo(f"Camera {camera_id} - Saved detection visualization to /tmp/")

# Debugging helper function
def debug_point_cloud_pipeline(realsense_inputs, detection_results):
    """
    Run comprehensive debugging on the point cloud generation pipeline
    """
    print("\n=== POINT CLOUD PIPELINE DEBUG ===")
    
    # Test transform function first
    check_transform_pointcloud_function()
    
    # Process each camera
    for camera_id, realsense_input in realsense_inputs.items():
        if camera_id not in detection_results:
            continue
            
        print(f"\n--- Camera {camera_id} ---")
        
        # Visualize detection box
        box_filter = detection_results[camera_id].box_filter[0]
        if hasattr(box_filter, 'numpy'):
            box_filter_np = box_filter.numpy()
        else:
            box_filter_np = np.array(box_filter)
            
        visualize_detection_box_on_image(
            realsense_input.color_image_np,
            realsense_input.depth_image_np,
            box_filter_np,
            camera_id
        )
        
        # Generate point cloud with detailed debugging
        pcd = get_single_pointcloud_fixed(realsense_input, detection_results[camera_id], camera_id)
        
        if pcd is not None:
            print(f"Camera {camera_id}: ✅ Successfully generated point cloud with {len(pcd.points)} points")
        else:
            print(f"Camera {camera_id}: ❌ Failed to generate point cloud")


def get_single_pointcloud_with_debug(realsense_input, groundingdino_output, camera_id=None):
    """Point cloud generation with extensive debugging"""
    
    try:
        print(f"\n=== DEBUGGING POINT CLOUD GENERATION FOR CAMERA {camera_id} ===")
        
        # Extract bounding box
        box_filter = groundingdino_output.box_filter[0]
        if hasattr(box_filter, 'numpy'):
            box_filter_np = box_filter.numpy() 
        else:
            box_filter_np = np.array(box_filter)
        
        center_x, center_y, width, height = box_filter_np
        print(f"Detection box: center=({center_x:.3f}, {center_y:.3f}), size=({width:.3f}, {height:.3f})")
        
        # Get image data
        color_image_np = realsense_input.color_image_np
        depth_image_np = realsense_input.depth_image_np
        camera_info = realsense_input.camera_info
        
        print(f"Image shapes: color={color_image_np.shape}, depth={depth_image_np.shape}")
        
        # Check depth image statistics
        depth_valid = depth_image_np[depth_image_np > 0]
        if len(depth_valid) > 0:
            print(f"Depth statistics: min={np.min(depth_valid):.3f}m, max={np.max(depth_valid):.3f}m, mean={np.mean(depth_valid):.3f}m")
            print(f"Valid depth pixels: {len(depth_valid)}/{depth_image_np.size} ({100*len(depth_valid)/depth_image_np.size:.1f}%)")
        else:
            print("ERROR: No valid depth values found!")
            return None
        
        # Get image dimensions
        height_img, width_img = depth_image_np.shape[:2]
        print(f"Image dimensions: {width_img}x{height_img}")
        
        # Calculate bounding box in pixels
        margin_w = width * 0.2
        margin_h = height * 0.2
        
        x_min = max(0, (center_x - width/2 - margin_w) * width_img)
        y_min = max(0, (center_y - height/2 - margin_h) * height_img)
        x_max = min(width_img, (center_x + width/2 + margin_w) * width_img)
        y_max = min(height_img, (center_y + height/2 + margin_h) * height_img)
        
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)
        
        print(f"Bounding box (pixels): ({x_min}, {y_min}) to ({x_max}, {y_max})")
        print(f"Box size: {x_max-x_min}x{y_max-y_min} pixels")
        
        # Check if bounding box is valid
        if x_max <= x_min or y_max <= y_min:
            print("ERROR: Invalid bounding box!")
            return None
        
        # Create mask
        mask = np.zeros((height_img, width_img), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        print(f"Mask pixels: {np.sum(mask)}")
        
        # Apply mask to depth
        masked_depth = np.copy(depth_image_np)
        masked_depth[~mask] = 0
        
        # Check masked depth statistics
        masked_valid = masked_depth[masked_depth > 0]
        if len(masked_valid) > 0:
            print(f"Masked depth: min={np.min(masked_valid):.3f}m, max={np.max(masked_valid):.3f}m, mean={np.mean(masked_valid):.3f}m")
            print(f"Valid masked pixels: {len(masked_valid)}")
        else:
            print("ERROR: No valid depth in masked region!")
            return None
        
        # Check intrinsic matrix
        intrinsic = camera_info["intrinsic_matrix"]
        print(f"Intrinsic matrix:\n{intrinsic}")
        print(f"Focal lengths: fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")
        print(f"Principal point: cx={intrinsic[0,2]:.1f}, cy={intrinsic[1,2]:.1f}")
        
        # Generate 3D points in camera frame
        print("\nGenerating 3D points...")
        xyz_camera = get_pointcloud(masked_depth, intrinsic)
        print(f"3D points shape: {xyz_camera.shape}")
        
        # Check camera frame points
        valid_camera = xyz_camera[:,:,2] > 0
        if np.sum(valid_camera) > 0:
            valid_camera_points = xyz_camera[valid_camera]
            print(f"Camera frame points: {len(valid_camera_points)}")
            print(f"X range: [{np.min(valid_camera_points[:,0]):.3f}, {np.max(valid_camera_points[:,0]):.3f}]")
            print(f"Y range: [{np.min(valid_camera_points[:,1]):.3f}, {np.max(valid_camera_points[:,1]):.3f}]")
            print(f"Z range: [{np.min(valid_camera_points[:,2]):.3f}, {np.max(valid_camera_points[:,2]):.3f}]")
        else:
            print("ERROR: No valid points in camera frame!")
            return None
        
        # Transform to robot frame
        print("\nTransforming to robot frame...")
        position = np.array(camera_info["position"])
        orientation = np.array(camera_info["orientation"])
        
        print(f"Camera position: {position}")
        print(f"Camera orientation: {orientation}")
        
        # Create transformation matrix
        r = R.from_quat(orientation)
        rotation_matrix = r.as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position
        
        print(f"Transform matrix:\n{transform}")
        
        # Apply transformation
        xyz_robot = transform_pointcloud(xyz_camera, transform)
        
        # Check robot frame points
        valid_robot = xyz_robot[:,:,2] > 0
        if np.sum(valid_robot) > 0:
            valid_robot_points = xyz_robot[valid_robot]
            print(f"Robot frame points: {len(valid_robot_points)}")
            print(f"X range: [{np.min(valid_robot_points[:,0]):.3f}, {np.max(valid_robot_points[:,0]):.3f}]")
            print(f"Y range: [{np.min(valid_robot_points[:,1]):.3f}, {np.max(valid_robot_points[:,1]):.3f}]")
            print(f"Z range: [{np.min(valid_robot_points[:,2]):.3f}, {np.max(valid_robot_points[:,2]):.3f}]")
            
            # Check if all points are at camera position (problem indicator)
            if np.allclose(valid_robot_points, position, atol=0.001):
                print("ERROR: All points are at camera position! Transformation issue detected.")
                return None
                
        else:
            print("ERROR: No valid points in robot frame!")
            return None
        
        # Extract valid points and colors
        points_flat = xyz_robot.reshape(-1, 3)
        mask_flat = (points_flat[:,2] > 0).reshape(-1)
        filtered_points = points_flat[mask_flat]
        
        color_flat = color_image_np.reshape(-1, 3)
        mask_2d_flat = mask.reshape(-1)
        filtered_colors = color_flat[mask_2d_flat & mask_flat]
        
        print(f"Final filtered points: {len(filtered_points)}")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        if len(filtered_colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)
        
        print("Point cloud created successfully!")

        points = np.asarray(pcd.points)

        # Fix 1: Downsample if too many points
        if len(points) > 50000:
            print(f"Downsampling from {len(points)} points...")
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            print(f"After downsampling: {len(pcd.points)} points")

        # Fix 2: Remove outliers
        print("Removing outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"After outlier removal: {len(pcd.points)} points")
        
        # Fix 3: Ensure point cloud is centered and scaled appropriately
        points = np.asarray(pcd.points)
        
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Scale to reasonable size for GraspNet (typical object size ~10-20cm)
        max_extent = np.max(np.abs(points_centered))
        if max_extent > 0.2:  # If object is larger than 20cm
            scale_factor = 0.15 / max_extent
            print(f"Scaling point cloud by {scale_factor:.3f}")
            points_scaled = points_centered * scale_factor + centroid
            
            pcd_fixed = o3d.geometry.PointCloud()
            pcd_fixed.points = o3d.utility.Vector3dVector(points_scaled)
            
            if pcd.has_colors():
                pcd_fixed.colors = pcd.colors
            if pcd.has_normals():
                pcd_fixed.normals = pcd.normals
            
            pcd = pcd_fixed
        
        # Fix 4: Ensure proper orientation (object on table)
        points = np.asarray(pcd.points)
        min_z = np.min(points[:, 2])
        
        # Move object to Z=0 (table level)
        if min_z > 0.01 or min_z < -0.01:
            print(f"Moving object to table level (offset: {-min_z:.3f})")
            pcd.translate([0, 0, -min_z])
        
        # Fix 5: Ensure normals exist and point outward
        if not pcd.has_normals():
            print("Computing normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=10)
        
        return pcd

        
    except Exception as e:
        print(f"ERROR in point cloud generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_fuse_pointcloud(realsense_input_dict, groundingdino_output_dict, frame_id=None):
    """
    Fuse the depth from all angles to a single pointcloud for an object
    using bounding boxes from each camera.
    
    Args:
        realsense_input_dict: Hashmap of realsense input(color, depth, camera_info)
        groundingdino_output_dict: Dictionary of bounding boxes for each camera (camera_id -> box_filter)
                         Each box_filter is [center_x, center_y, width, height]
        frame_id: Optional reference frame ID (not required for this approach)
    Returns:
        fuse_pcd
    """
    try:
        pcds = []
        transformations = {}
        camera_info_dict = {}

        # Verify camera transforms if possible
        try:
            verify_camera_transform()
        except Exception as e:
            rospy.logwarn(f"Camera transform verification skipped: {e}")
        
        # Process each camera with its own bounding box
        for camera_id, realsense_input in realsense_input_dict.items():
            # Skip if no bounding box for this camera
            if camera_id not in groundingdino_output_dict:
                rospy.loginfo(f"No bounding box for camera {camera_id}, skipping")
                continue
                
            box_filter = groundingdino_output_dict[camera_id].box_filter[0]
            rospy.loginfo(f"Camera {camera_id} - Box Filter: {box_filter}")

            # Convert box_filter to numpy if it's a tensor
            if hasattr(box_filter, 'numpy'):
                    box_filter_np = box_filter.numpy() 
            else:
                box_filter_np = np.array(box_filter)
                    
            # Extract box coordinates (center_x, center_y, width, height)
            center_x, center_y, width, height = box_filter_np
            
            # Debug the bounding box values
            rospy.loginfo(f"Camera {camera_id} - Detection: center=({center_x:.3f}, {center_y:.3f}), size=({width:.3f}, {height:.3f})")
            
            # Validate bounding box
            if width <= 0 or height <= 0:
                rospy.logerr(f"Camera {camera_id} - Invalid bounding box dimensions: width={width}, height={height}")
                return None
                
            if center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1:
                rospy.logerr(f"Camera {camera_id} - Invalid bounding box center: ({center_x}, {center_y})")
                return None
            
            # Get camera data
            color_image_np = realsense_input.color_image_np
            depth_image_np = realsense_input.depth_image_np
            camera_info = realsense_input.camera_info

            # Store camera info for later use
            camera_info_dict[camera_id] = camera_info
            
            # Get image dimensions
            height_img, width_img = depth_image_np.shape[:2] if len(depth_image_np.shape) > 2 else depth_image_np.shape
            rospy.loginfo(f"Camera {camera_id} - Image dimensions: {width_img}x{height_img}")
            
            # FIX 1: Increase margin for better object capture
            margin_w = width * 0.2  # (20% margin)
            margin_h = height * 0.2  # (20% margin)
            
            # Calculate pixel coordinates from normalized coordinates with margin
            x_min = max(0, (center_x - width/2 - margin_w) * width_img)
            y_min = max(0, (center_y - height/2 - margin_h) * height_img)
            x_max = min(width_img, (center_x + width/2 + margin_w) * width_img)
            y_max = min(height_img, (center_y + height/2 + margin_h) * height_img)
            
            # Convert to integers
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)
            
            rospy.loginfo(f"Camera {camera_id} - Bounding box in pixels: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            
            # FIX 2: Validate pixel coordinates
            if x_max <= x_min or y_max <= y_min:
                rospy.logerr(f"Camera {camera_id} - Invalid pixel bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                return None
            
            # Check depth image statistics before masking
            depth_valid = depth_image_np[depth_image_np > 0]
            if len(depth_valid) == 0:
                rospy.logerr(f"Camera {camera_id} - No valid depth data in entire image")
                return None
                
            rospy.loginfo(f"Camera {camera_id} - Depth range: {np.min(depth_valid):.3f} to {np.max(depth_valid):.3f}m")
            
            # Create a mask for the pixels within the bounding box
            mask = np.zeros((height_img, width_img), dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            rospy.loginfo(f"Camera {camera_id} - Mask covers {np.sum(mask)} pixels ({100*np.sum(mask)/(width_img*height_img):.1f}% of image)")
            
            # Apply mask to depth image
            masked_depth = np.copy(depth_image_np)
            masked_depth[~mask] = 0
            
            # FIX 3: Check masked depth statistics
            masked_valid = masked_depth[masked_depth > 0]
            if len(masked_valid) == 0:
                rospy.logerr(f"Camera {camera_id} - No valid depth data in bounding box region")
                
                # Debug: Save depth image for inspection
                import cv2
                debug_depth = (depth_image_np / np.max(depth_image_np) * 255).astype(np.uint8)
                debug_depth_colored = cv2.applyColorMap(debug_depth, cv2.COLORMAP_JET)
                cv2.rectangle(debug_depth_colored, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imwrite(f'/tmp/debug_depth_cam_{camera_id}.png', debug_depth_colored)
                rospy.loginfo(f"Camera {camera_id} - Saved debug depth image to /tmp/debug_depth_cam_{camera_id}.png")
                
                continue  # Skip this camera
            
            rospy.loginfo(f"Camera {camera_id} - Masked depth range: {np.min(masked_valid):.3f} to {np.max(masked_valid):.3f}m, mean: {np.mean(masked_valid):.3f}m")
            rospy.loginfo(f"Camera {camera_id} - Valid masked pixels: {len(masked_valid)}")
                
            # Convert masked depth to point cloud
            xyz = get_pointcloud(masked_depth, camera_info["intrinsic_matrix"])
            
            # FIX 4: Debug camera frame points before transformation
            valid_camera_mask = xyz[:,:,2] > 0
            if np.sum(valid_camera_mask) == 0:
                rospy.logerr(f"Camera {camera_id} - No valid 3D points generated from depth")
                return None
                
            valid_camera_points = xyz[valid_camera_mask]
            rospy.loginfo(f"Camera {camera_id} - Camera frame points: {len(valid_camera_points)}")
            rospy.loginfo(f"Camera {camera_id} - Camera frame X: [{np.min(valid_camera_points[:,0]):.3f}, {np.max(valid_camera_points[:,0]):.3f}]")
            rospy.loginfo(f"Camera {camera_id} - Camera frame Y: [{np.min(valid_camera_points[:,1]):.3f}, {np.max(valid_camera_points[:,1]):.3f}]")
            rospy.loginfo(f"Camera {camera_id} - Camera frame Z: [{np.min(valid_camera_points[:,2]):.3f}, {np.max(valid_camera_points[:,2]):.3f}]")
            
            # Apply transform to world coordinates
            position = np.array(camera_info["position"]).reshape(3, 1)
            orientation = np.array(camera_info["orientation"])  # [x, y, z, w]
            
            # FIX 5: Use scipy for quaternion conversion instead of pybullet
            from scipy.spatial.transform import Rotation as R
            rotation = R.from_quat(orientation).as_matrix()
            
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = position.flatten()
            
            rospy.loginfo(f"Camera {camera_id} - Transform:\nPosition: {position.flatten()}\nRotation matrix:\n{rotation}")
            
            # Transform points to world coordinates
            transformed_points = transform_pointcloud(xyz, transform)
            
            # FIX 7: Improved point filtering
            # Reshape for filtering
            points_shape = transformed_points.shape
            points_reshaped = transformed_points.reshape(-1, 3)
            mask_reshaped = mask.reshape(-1)
            
            # Create combined mask: bounding box AND reasonable Z values
            reasonable_z_mask = (points_reshaped[:,2] > -0.1) & (points_reshaped[:,2] < 2.0)  # Between -10cm and 2.0m
            combined_mask = mask_reshaped & reasonable_z_mask
            
            # Filter points using the combined mask
            filtered_points = points_reshaped[combined_mask]
            
            # Get corresponding colors
            color_reshaped = color_image_np.reshape(-1, 3)
            filtered_colors = color_reshaped[combined_mask]
            
            # FIX 8: Check filtering results
            if len(filtered_points) == 0:
                rospy.logerr(f"Camera {camera_id} - No points after filtering")
                rospy.loginfo(f"Camera {camera_id} - Mask sum: {np.sum(mask_reshaped)}")
                rospy.loginfo(f"Camera {camera_id} - Reasonable Z sum: {np.sum(reasonable_z_mask)}")
                rospy.loginfo(f"Camera {camera_id} - Combined mask sum: {np.sum(combined_mask)}")
                return None
            
            # Log statistics
            x_values = filtered_points[:, 0]
            y_values = filtered_points[:, 1]
            z_values = filtered_points[:, 2]
            
            rospy.loginfo(f"Camera {camera_id} - Filtered points stats:")
            rospy.loginfo(f"X: max={np.max(x_values):.3f}, min={np.min(x_values):.3f}, mean={np.mean(x_values):.3f}")
            rospy.loginfo(f"Y: max={np.max(y_values):.3f}, min={np.min(y_values):.3f}, mean={np.mean(y_values):.3f}")
            rospy.loginfo(f"Z: max={np.max(z_values):.3f}, min={np.min(z_values):.3f}, mean={np.mean(z_values):.3f}")
            rospy.loginfo(f"Total valid points: {len(filtered_points)}")
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)
            
            # FIX 9: More conservative outlier removal
            if len(filtered_points) > 100:
                original_count = len(pcd.points)
                pcd, outlier_indices = pcd.remove_statistical_outlier(reconstruction_config['nb_neighbors'], reconstruction_config['std_ratio'])  # Increased threshold
                rospy.loginfo(f"Camera {camera_id} - After outlier removal: {len(pcd.points)} points (removed {original_count - len(pcd.points)})")
            
            # FIX 10: Optional voxel downsampling (only if too many points)
            if len(pcd.points) > 50000:  # Only downsample if too many points
                original_count = len(pcd.points)
                pcd = pcd.voxel_down_sample(reconstruction_config['voxel_size'])
                rospy.loginfo(f"Camera {camera_id} - After downsampling: {len(pcd.points)} points (removed {original_count - len(pcd.points)})")
            
            # FIX 11: Ensure we still have a reasonable number of points
            if len(pcd.points) < 100:
                rospy.logwarn(f"Camera {camera_id} - Too few points after processing: {len(pcd.points)}")
                continue

            # Check for initial plane detection in individual views
            try:
                plane_model, inliers, success = detect_table_plane(pcd, distance_threshold=reconstruction_config.get('plane_distance_threshold', 0.01))
                if success:
                    normal = np.array(plane_model[:3])
                    normal = normal / np.linalg.norm(normal)
                    rospy.loginfo(f"Camera {camera_id} - Detected plane normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                    
                    # Check if plane is roughly horizontal (in world coordinates)
                    angle = np.arccos(np.abs(normal[2])) * 180 / np.pi
                    rospy.loginfo(f"Camera {camera_id} - Detected plane angle with vertical: {angle:.1f}°")
            except Exception as e:
                rospy.logwarn(f"Camera {camera_id} - Plane detection failed: {e}")

            # visualization
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            o3d.visualization.draw_geometries([pcd, frame], f"Camera {camera_id} - Point Cloud")
            
            pcds.append(pcd)
            rospy.loginfo(f"Camera {camera_id} - Added point cloud with {len(pcd.points)} points")
        
        # Process point clouds for fusion
        if len(pcds) == 0:
            rospy.logwarn("No valid point clouds to merge")
            return [], [], [], []
        elif len(pcds) == 1:
            rospy.loginfo("Only one valid point cloud, no fusion needed")
            return pcds[0]
        else:
            # Use process_pcds function to align and merge the point clouds
            fused_trans_world, fused_pcd_world = process_pcds(pcds, reconstruction_config)
            
            if fused_pcd_world is not None and len(fused_pcd_world.points) > 0:
                rospy.loginfo(f"Successfully fused {len(pcds)} point clouds, resulting in {len(fused_pcd_world.points)} points")

                # After fusion is complete, add orientation correction
                if reconstruction_config.get('orient_for_grasping', False):
                    print("Orienting point cloud for top-down grasping...")
                    fused_pcd_canonical, fused_trans_canonical = orient_for_top_grasping(
                        fused_pcd_world, 
                        debug=reconstruction_config.get('debug_orientation', False)
                    )
                    # Optional visualization of final result
                    if reconstruction_config.get('visualize_final', False):
                        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                        o3d.visualization.draw_geometries([fused_pcd_canonical, frame], "Final Aligned Point Cloud")
                return fused_pcd_world, fused_pcd_canonical, fused_trans_world, fused_trans_canonical
            else:
                rospy.logwarn("Fusion resulted in empty point cloud")
                return [], [], [], []

        # rospy.loginfo("Performing explicit table alignment...")
        # aligned_pcd, alignment_transform, alignment_success = align_to_table_plane(
        #     fused_pcd_world,
        #     distance_threshold=reconstruction_config.get('plane_distance_threshold', 0.01),
        #     visualize=reconstruction_config.get('visualize_alignment', False)
        # )

        # if not alignment_success:
        #     rospy.logwarn("Table alignment failed, falling back to orient_for_top_grasping")
        #     # Fall back to orient_for_top_grasping if table alignment fails
        #     if reconstruction_config.get('orient_for_grasping', True):
        #         aligned_pcd, alignment_transform = orient_for_top_grasping(
        #             fused_pcd_world, 
        #             debug=reconstruction_config.get('debug_orientation', False)
        #         )
        
        # # Post-process to clean up point cloud
        # fused_pcd_canonical = post_process_aligned_pointcloud(
        #     aligned_pcd,
        #     min_height=reconstruction_config.get('min_object_height', 0.005),
        #     max_height=reconstruction_config.get('max_object_height', 0.5),
        #     voxel_size=reconstruction_config.get('final_voxel_size', 0.002)
        # )
        
        # # Calculate canonical transform
        # fused_trans_canonical = alignment_transform
        
        # # Optional visualization of final result
        # if reconstruction_config.get('visualize_final', False):
        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        #     o3d.visualization.draw_geometries([fused_pcd_canonical, frame], "Final Aligned Point Cloud")
        
        # return fused_pcd_world, fused_pcd_canonical, fused_trans_world, fused_trans_canonical
                
    except Exception as e:
        rospy.logerr(f"Failed to create single fuse cloudpoint: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        return None

def post_process_aligned_pointcloud(pcd, min_height=0.005, max_height=0.5, voxel_size=0.002):
    """
    Post-process the aligned point cloud:
    1. Remove points below the table
    2. Remove points too far above the table
    3. Optional voxel downsampling
    
    Args:
        pcd: Open3D point cloud (already aligned)
        min_height: Minimum height above detected table to keep points
        max_height: Maximum height above detected table to keep points
        voxel_size: Voxel size for downsampling (or None for no downsampling)
        
    Returns:
        processed_pcd: Post-processed point cloud
    """
    if len(pcd.points) == 0:
        return pcd
    
    # Make a copy
    processed_pcd = copy.deepcopy(pcd)
    
    # Get points as numpy array
    points = np.asarray(processed_pcd.points)
    
    # Get colors
    has_colors = processed_pcd.has_colors()
    if has_colors:
        colors = np.asarray(processed_pcd.colors)
    
    # Find lowest z-value (approximate table height)
    z_values = points[:, 2]
    if len(z_values) == 0:
        return processed_pcd
    
    min_z = np.min(z_values)
    
    # Create height mask
    height_above_table = z_values - min_z
    height_mask = (height_above_table >= min_height) & (height_above_table <= max_height)
    
    # Apply mask to points
    filtered_points = points[height_mask]
    
    # Recreate point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Apply mask to colors too if they exist
    if has_colors:
        filtered_colors = colors[height_mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # Apply statistical outlier removal if we have enough points
    if len(filtered_points) > 100:
        filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
    
    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0 and len(filtered_pcd.points) > 5000:
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size)
    
    rospy.loginfo(f"Post-processing: original={len(points)} points, filtered={len(filtered_pcd.points)} points")
    
    return filtered_pcd

def verify_camera_transform():
    """
    Utility function to verify camera transform quality.
    Useful for diagnosing issues with eye-in-hand calibration.
    """
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    rospy.loginfo("=== VERIFYING CAMERA TRANSFORM ===")
    
    # Get camera to base transform
    try:
        transform = tfBuffer.lookup_transform('base_link', 'camera_color_optical_frame', 
                                             rospy.Time(0), rospy.Duration(5.0))
        
        # Extract rotation as matrix
        quat = [transform.transform.rotation.x, transform.transform.rotation.y,
                transform.transform.rotation.z, transform.transform.rotation.w]
        r = R.from_quat(quat)
        rot_matrix = r.as_matrix()
        
        # Check camera Z-axis (should point along camera's viewing direction)
        camera_z = rot_matrix[:, 2]
        angle_to_down = np.arccos(-camera_z[2]) * 180 / np.pi  # Angle between camera z and world -z
        
        rospy.loginfo(f"Camera position: [{transform.transform.translation.x:.3f}, "
                     f"{transform.transform.translation.y:.3f}, "
                     f"{transform.transform.translation.z:.3f}]")
        rospy.loginfo(f"Camera z-axis: [{camera_z[0]:.3f}, {camera_z[1]:.3f}, {camera_z[2]:.3f}]")
        rospy.loginfo(f"Angle between camera view direction and downward: {angle_to_down:.1f}°")
        
        # Print full transformation matrix for debugging
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = [transform.transform.translation.x, 
                   transform.transform.translation.y, 
                   transform.transform.translation.z]
        
        rospy.loginfo("Full transformation matrix (camera to base):")
        rospy.loginfo(f"{T[0,0]:.3f} {T[0,1]:.3f} {T[0,2]:.3f} {T[0,3]:.3f}")
        rospy.loginfo(f"{T[1,0]:.3f} {T[1,1]:.3f} {T[1,2]:.3f} {T[1,3]:.3f}")
        rospy.loginfo(f"{T[2,0]:.3f} {T[2,1]:.3f} {T[2,2]:.3f} {T[2,3]:.3f}")
        rospy.loginfo(f"{T[3,0]:.3f} {T[3,1]:.3f} {T[3,2]:.3f} {T[3,3]:.3f}")
        
        return T, angle_to_down
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Transform verification failed: {e}")
        return None, None

def orient_pointcloud_back_to_world(pcd, transformation_matrix):
    """
    Reverts the oriented point cloud back to world coordinates.
    
    Args:
        pcd: Open3D point cloud that has been oriented for grasping
        transformation_matrix: Transformation matrix that was used to orient the point cloud
    
    Returns:
        Reoriented point cloud in world coordinates
    """
    # Compute the inverse of the transformation matrix
    inverse_transform = np.linalg.inv(transformation_matrix)
    
    # Apply the inverse transformation to revert the point cloud back to world coordinates
    pcd.transform(inverse_transform)
    
    return pcd

def detect_and_correct_orientation(pcd, method='pca'):
    """
    Detect the orientation of the point cloud and correct it to face upward.
    
    Args:
        pcd: Open3D point cloud
        method: 'pca' for PCA-based, 'ransac' for plane-fitting based
    Returns:
        corrected_pcd: Reoriented point cloud
        transform: Applied transformation matrix
    """
    corrected_pcd = copy.deepcopy(pcd)
    
    if method == 'pca':
        # Use PCA to find the principal axes
        points = np.asarray(corrected_pcd.points)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Compute covariance matrix and eigenvalues/eigenvectors
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues (largest to smallest)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # The smallest eigenvector is often the "up" direction for flat objects
        # We want this to align with the Z-axis
        up_vector = eigenvectors[:, 2]
        
        # If the up vector points downward, flip it
        if up_vector[2] < 0:
            up_vector = -up_vector
        
        # Create rotation matrix to align up_vector with Z-axis
        z_axis = np.array([0, 0, 1])
        v = np.cross(up_vector, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(up_vector, z_axis)
        
        if s < 1e-6:  # Vectors are already aligned
            rotation = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            rotation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -np.dot(rotation, centroid) + centroid
        
    elif method == 'ransac':
        # Use RANSAC to fit a plane and align the object
        plane_model, inliers = corrected_pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
        
        # Extract plane normal
        normal = np.array(plane_model[:3])
        
        # Ensure normal points upward
        if normal[2] < 0:
            normal = -normal
        
        # Create rotation matrix to align normal with Z-axis
        z_axis = np.array([0, 0, 1])
        v = np.cross(normal, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(normal, z_axis)
        
        if s < 1e-6:  # Vectors are already aligned
            rotation = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            rotation = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        
        # Center the point cloud
        points = np.asarray(corrected_pcd.points)
        centroid = np.mean(points, axis=0)
        transform[:3, 3] = -np.dot(rotation, centroid) + centroid
    
    # Apply transformation
    corrected_pcd.transform(transform)
    
    return corrected_pcd, transform


def detect_object_top_surface(pcd, percentile=90):
    """
    Detect the top surface of an object by analyzing point distribution.
    
    Args:
        pcd: Open3D point cloud
        percentile: Percentile of points to consider as "top"
    Returns:
        top_normal: Normal vector of the top surface
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Find points in the top percentile by Z coordinate
    z_threshold = np.percentile(points[:, 2], percentile)
    top_indices = points[:, 2] >= z_threshold
    
    if np.sum(top_indices) < 10:
        # Fallback to simpler method
        return np.array([0, 0, 1])
    
    # Average the normals of top points
    top_normals = normals[top_indices]
    avg_normal = np.mean(top_normals, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    
    # Ensure it points upward
    if avg_normal[2] < 0:
        avg_normal = -avg_normal
    
    return avg_normal


# def orient_for_top_grasping(pcd, debug=False):
#     """
#     Enhanced version of orient_for_top_grasping specifically for eye-in-hand setups.
    
#     Args:
#         pcd: Open3D point cloud
#         debug: If True, visualize the orientation process
#     Returns:
#         oriented_pcd: Properly oriented point cloud
#         transform: Applied transformation
#     """
#     # Start with basic orientation correction
#     oriented_pcd, transform1 = detect_and_correct_orientation(pcd, method='pca')
    
#     # Try to detect table plane
#     plane_model, inliers, success = detect_table_plane(oriented_pcd)
    
#     if success:
#         # Use detected plane for final alignment
#         aligned_pcd, table_transform, _ = align_to_table_plane(oriented_pcd, plane_model)
#         transform = table_transform @ transform1
#         oriented_pcd = aligned_pcd
#     else:
#         # Fall back to original method if plane detection fails
#         oriented_pcd.estimate_normals()
#         top_normal = detect_object_top_surface(oriented_pcd)
        
#         # Fine-tune orientation based on top surface normal
#         z_axis = np.array([0, 0, 1])
#         angle = np.arccos(np.clip(np.dot(top_normal, z_axis), -1, 1))
        
#         if angle > np.radians(10):  # If deviation is significant
#             # Compute rotation axis
#             rotation_axis = np.cross(top_normal, z_axis)
#             rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
#             # Create rotation matrix using Rodrigues' formula
#             K = np.array([
#                 [0, -rotation_axis[2], rotation_axis[1]],
#                 [rotation_axis[2], 0, -rotation_axis[0]],
#                 [-rotation_axis[1], rotation_axis[0], 0]
#             ])
            
#             rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
#             # Apply fine-tuning transformation
#             transform2 = np.eye(4)
#             transform2[:3, :3] = rotation
            
#             oriented_pcd.transform(transform2)
#             transform = transform2 @ transform1
#         else:
#             transform = transform1
    
#     # Ensure the object is centered and sitting on a plane
#     points = np.asarray(oriented_pcd.points)
#     min_z = np.min(points[:, 2])
#     centroid = np.mean(points, axis=0)
    
#     # Translate to center horizontally and place on ground plane
#     translation = np.eye(4)
#     translation[:3, 3] = [-centroid[0], -centroid[1], -min_z]
    
#     oriented_pcd.transform(translation)
#     transform = translation @ transform
    
#     if debug:
#         # Visualize the orientation process
#         coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#         o3d.visualization.draw_geometries(
#             [oriented_pcd, coord_frame],
#             window_name="Oriented Point Cloud for Top Grasping"
#         )
    
#     return oriented_pcd, transform

def orient_for_top_grasping(pcd, debug=False):
    """
    Correct the orientation of the object without changing its position.
    The object will be aligned upright but stay in the same world position.
    
    Args:
        pcd: Open3D point cloud of the object
        debug: If True, visualize the orientation process
    Returns:
        oriented_pcd: Properly oriented point cloud, same position
        transform: Applied transformation matrix (rotation only)
    """
    # Step 1: Initial orientation correction (using PCA or RANSAC)
    oriented_pcd, transform1 = detect_and_correct_orientation(pcd, method='pca')

    # Step 2: Detect the object’s top surface normal (the upward direction)
    oriented_pcd.estimate_normals()
    top_normal = detect_object_top_surface(oriented_pcd)
    
    # Step 3: Fine-tune orientation based on top surface normal
    z_axis = np.array([0, 0, 1])  # World 'up' direction

    # Calculate the angle between the current top normal and the desired vertical direction
    angle = np.arccos(np.clip(np.dot(top_normal, z_axis), -1, 1))

    if angle > np.radians(10):  # If the object is tilted significantly
        # Compute the rotation axis to align the top normal with the Z-axis
        rotation_axis = np.cross(top_normal, z_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Create a rotation matrix using Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])

        # Final rotation matrix
        rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # Apply the rotation to the object, but **do not translate it**
        transform2 = np.eye(4)
        transform2[:3, :3] = rotation
        
        # Apply rotation without changing the position of the object
        oriented_pcd.transform(transform2)
        transform = transform2 @ transform1
    else:
        # No rotation needed if angle is small (within 10 degrees)
        transform = transform1
    
    # Step 4: Visualize (optional)
    if debug:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            [oriented_pcd, coord_frame],
            window_name="Oriented Point Cloud for Top Grasping"
        )
    
    return oriented_pcd, transform

def detect_table_plane(pcd, distance_threshold=0.01, min_points=500):
    """
    Detect the table plane in a point cloud using RANSAC
    
    Args:
        pcd: Open3D point cloud
        distance_threshold: RANSAC distance threshold
        min_points: Minimum number of points for valid plane
        
    Returns:
        plane_model: (a, b, c, d) plane equation
        inliers: Indices of inlier points
        success: Whether plane detection was successful
    """
    if len(pcd.points) < min_points:
        return None, None, False
    
    # Make a copy to avoid modifying original
    pcd_copy = copy.deepcopy(pcd)
    
    # Estimate normals if they don't exist
    if not pcd_copy.has_normals():
        pcd_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
    
    # Use RANSAC to find dominant plane
    try:
        plane_model, inliers = pcd_copy.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        # Check if we have enough inliers
        if len(inliers) < min_points:
            rospy.logwarn(f"Not enough inliers for plane: {len(inliers)} < {min_points}")
            return plane_model, inliers, False
        
        # Verify plane is roughly horizontal (normal pointing up)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        
        # Angle between normal and vertical
        angle_with_vertical = np.arccos(np.abs(normal[2])) * 180 / np.pi
        
        if angle_with_vertical > 25:  # More than 25° from vertical
            rospy.logwarn(f"Detected plane is not horizontal: {angle_with_vertical:.1f}° from vertical")
            rospy.logwarn(f"Normal: [{a:.3f}, {b:.3f}, {c:.3f}]")
            return plane_model, inliers, False
        
        # Success
        return plane_model, inliers, True
        
    except Exception as e:
        rospy.logerr(f"Plane detection failed: {e}")
        return None, None, False

def align_to_table_plane(pcd, plane_model=None, distance_threshold=0.01, visualize=False):
    """
    Align point cloud to make table plane horizontal
    
    Args:
        pcd: Open3D point cloud
        plane_model: Optional pre-computed plane model. If None, will be detected
        distance_threshold: RANSAC threshold for plane detection
        visualize: If True, visualize before and after alignment
        
    Returns:
        aligned_pcd: Point cloud aligned to table
        transform: Applied transformation
        success: Whether alignment was successful
    """
    # Make a copy of the point cloud
    aligned_pcd = copy.deepcopy(pcd)
    
    # Detect plane if not provided
    if plane_model is None:
        plane_model, inliers, success = detect_table_plane(aligned_pcd, distance_threshold)
        if not success:
            rospy.logwarn("Could not detect stable table plane. Skipping alignment.")
            return aligned_pcd, np.eye(4), False
    
    # Extract table normal (a, b, c from ax + by + cz + d = 0)
    a, b, c, d = plane_model
    table_normal = np.array([a, b, c])
    
    # Make sure normal points upward (positive z)
    if table_normal[2] < 0:
        table_normal = -table_normal
        d = -d
    
    # Normalize the normal vector
    table_normal = table_normal / np.linalg.norm(table_normal)
    
    # Create rotation to align table normal with z-axis [0,0,1]
    z_axis = np.array([0, 0, 1])
    
    # Compute angle between normals
    dot_product = np.dot(table_normal, z_axis)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # If normals are already aligned (within 1 degree), no rotation needed
    if angle < np.radians(1.0):
        rospy.loginfo("Table already horizontal (within 1°). No alignment needed.")
        return aligned_pcd, np.eye(4), True
    
    # Compute rotation axis (cross product)
    rotation_axis = np.cross(table_normal, z_axis)
    
    # Handle case when vectors are parallel or anti-parallel
    if np.linalg.norm(rotation_axis) < 1e-6:
        if dot_product > 0:  # Already aligned
            return aligned_pcd, np.eye(4), True
        else:  # Anti-parallel, rotate around x-axis
            rotation_axis = np.array([1, 0, 0])
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Create rotation matrix (Rodrigues' formula)
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    
    rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Get centroid of point cloud
    points = np.asarray(aligned_pcd.points)
    centroid = np.mean(points, axis=0)
    
    # Create full transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    
    # Apply rotation around centroid
    transform[:3, 3] = centroid - np.dot(rotation, centroid)
    
    # Apply transformation
    aligned_pcd.transform(transform)
    
    # Optional: visualize before and after
    if visualize:
        # Create coordinate frames
        original_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        aligned_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Show original
        o3d.visualization.draw_geometries([pcd, original_frame], "Original Point Cloud")
        
        # Show aligned
        o3d.visualization.draw_geometries([aligned_pcd, aligned_frame], "Aligned Point Cloud")
    
    # Verify the alignment result
    new_plane_model, new_inliers, new_success = detect_table_plane(aligned_pcd, distance_threshold)
    if new_success:
        new_normal = np.array(new_plane_model[:3])
        new_normal = new_normal / np.linalg.norm(new_normal)
        new_angle = np.arccos(np.abs(new_normal[2])) * 180 / np.pi
        rospy.loginfo(f"After alignment: table normal = [{new_normal[0]:.3f}, {new_normal[1]:.3f}, {new_normal[2]:.3f}]")
        rospy.loginfo(f"After alignment: angle with vertical = {new_angle:.2f}°")
    
    return aligned_pcd, transform, True