import open3d as o3d
import numpy as np
import os
import copy
from typing import List, Tuple
import cv2


class RGBDStitcher:
    def __init__(self, intrinsic: o3d.camera.PinholeCameraIntrinsic):
        """
        Initialize RGBD stitching pipeline
        Args:
            intrinsic: Camera intrinsic parameters
        """
        self.intrinsic = intrinsic
        self.voxel_size = 0.02  # Default voxel size for downsampling
        self.distance_threshold = 0.05  # Default distance threshold for registration
        self.optimization_modulus = 2

    def create_point_cloud_from_rgbd(
            self,
            color_img: np.ndarray,
            depth_img: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Create point cloud from RGBD image pair
        Args:
            color_img: RGB image as numpy array
            depth_img: Depth image as numpy array
        Returns:
            Point cloud object
        """
        # Convert images to Open3D format
        color_o3d = o3d.geometry.Image(color_img)
        depth_o3d = o3d.geometry.Image(depth_img)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,  # Typical scale for depth in millimeters
            depth_trunc=3.0,     # Max depth in meters
            convert_rgb_to_intensity=False)

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.intrinsic)

        return pcd

    def preprocess_point_cloud(
            self,
            pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud,
                                                   o3d.pipelines.registration.Feature]:
        """
        Preprocess point cloud for registration
        """
        # Downsample
        pcd_down = pcd.voxel_down_sample(self.voxel_size)

        # Estimate normals
        radius_normal = self.voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute FPFH features
        radius_feature = self.voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, fpfh

    def register_point_clouds(
            self,
            source: o3d.geometry.PointCloud,
            target: o3d.geometry.PointCloud,
            initial_transform=np.identity(4)) -> Tuple[np.ndarray, float]:
        """
        Register two point clouds using color and geometry information
        Args:
            source: Source point cloud
            target: Target point cloud
            initial_transform: Initial transformation guess
        Returns:
            transformation_matrix: Final transformation matrix
            fitness: Registration fitness score
        """
        # Preprocess point clouds
        source_down, source_fpfh = self.preprocess_point_cloud(source)
        target_down, target_fpfh = self.preprocess_point_cloud(target)

        # Color-based registration (if colors available)
        if source.has_colors() and target.has_colors():
            result_color = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down,
                self.distance_threshold,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=50))
            initial_transform = result_color.transformation

        # Geometry-based refinement
        result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down,
                self.distance_threshold,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

        return result_icp.transformation, result_icp.fitness

    def stitch_sequence(self, color_images: List[np.ndarray], depth_images: List[np.ndarray]) -> o3d.geometry.PointCloud:
        """
        Stitch a sequence of RGBD images into a single point cloud
        """
        if len(color_images) != len(depth_images):
            raise ValueError("Number of color and depth images must match")

        # Create the first point cloud
        combined_cloud = self.create_point_cloud_from_rgbd(color_images[0], depth_images[0])

        global_transform = np.identity(4)

        # Process each subsequent image
        for i in range(1, len(color_images)):
            # Create point cloud from the current RGBD pair
            current_cloud = self.create_point_cloud_from_rgbd(color_images[i], depth_images[i])

            # Register the current cloud to the combined cloud
            transform, fitness = self.register_point_clouds(current_cloud, combined_cloud)

            # Transform and combine point clouds
            current_cloud.transform(transform)
            combined_cloud += current_cloud

            # Optional: Cleanup and optimization
            if i % self.optimization_modulus == 0:  # Every N frames
                # Check the type and size of combined cloud
                print(f"Combined cloud type: {type(combined_cloud)}")
                print(f"Number of points: {len(combined_cloud.points)}")

                # Remove invalid points before downsampling
                points = np.asarray(combined_cloud.points)
                valid_points = points[~np.isnan(points).any(axis=1)]
                valid_cloud = o3d.geometry.PointCloud()
                valid_cloud.points = o3d.utility.Vector3dVector(valid_points)

                # Now apply voxel downsampling
                combined_cloud = combined_cloud.voxel_down_sample(self.voxel_size)

                if len(combined_cloud.points) == 0:
                    print("Warning: Combined cloud empty after downsampling.")

                # Optionally: Remove noise
                if len(combined_cloud.points) > 1000:
                    combined_cloud, _ = combined_cloud.remove_statistical_outlier(
                            nb_neighbors=20, std_ratio=2.0)
                else:
                     print("Skipping outlier removal due to small point cloud size.")

                if len(combined_cloud.points) == 0:
                    print("Warning: Combined cloud empty after removing outliers.")

        return combined_cloud

    def visualize_registration(
            self,
            source: o3d.geometry.PointCloud,
            target: o3d.geometry.PointCloud,
            transformed: o3d.geometry.PointCloud = None):
        """
        Visualize registration results
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add geometries with different colors
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        if not source.has_colors():
            source_temp.paint_uniform_color([1, 0, 0])  # Red
        if not target.has_colors():
            target_temp.paint_uniform_color([0, 1, 0])  # Green

        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)
        if transformed is not None:
            transformed_temp = copy.deepcopy(transformed)
            if not transformed.has_colors():
                transformed_temp.paint_uniform_color([0, 0, 1])  # Blue
            vis.add_geometry(transformed_temp)

        # Set viewing parameters
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()

    def load_default(self):
        return self.load_dataset_two_folders(
                './camera',
                'rgb',
                'depth')

    def load_dataset_two_folders(
            self,
            folder_path,
            rgb_foldername,
            depth_foldername):
        rgb_images = []
        depth_images = []

        rgb_folder = os.path.join(folder_path, rgb_foldername)
        for filename in sorted(os.listdir(rgb_folder)):
            image_path = os.path.join(rgb_folder, filename)
            rgb_image = cv2.imread(image_path)
            rgb_images.append(rgb_image)

        depth_folder = os.path.join(folder_path, depth_foldername)
        for filename in sorted(os.listdir(depth_folder)):
            image_path = os.path.join(depth_folder, filename)
            depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            depth_images.append(depth_image)

        if len(rgb_images) % self.optimization_modulus != 0:
            for i in range(len(rgb_images) % self.optimization_modulus):
                rgb_images.pop()
                depth_images.pop()

        return rgb_images, depth_images


# Example usage
def main():
    # Initialize camera intrinsic parameters (example for Intel RealSense D435)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480,
        fx=615.6707153320312, fy=615.962158203125,
        cx=326.0557861328125, cy=240.55592346191406)

    # Create stitcher
    stitcher = RGBDStitcher(intrinsic)

    # Example: Load and process a sequence of RGBD images
    color_images = []  # List of your color images as numpy arrays
    depth_images = []  # List of your depth images as numpy arrays
    # Load your RGBD images here
    # Example:
    # for i in range(num_frames):
    #     color = cv2.imread(f"color_{i}.png")
    #     depth = cv2.imread(f"depth_{i}.png", cv2.IMREAD_UNCHANGED)
    #     color_images.append(color)
    #     depth_images.append(depth)

    color_images, depth_images = stitcher.load_default()

    # Perform stitching
    combined_cloud = stitcher.stitch_sequence(color_images, depth_images)

    # Visualize result
    o3d.visualization.draw_geometries([combined_cloud])


if __name__ == "__main__":
    main()
