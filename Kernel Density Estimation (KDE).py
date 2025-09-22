# -*- coding: utf-8 -*-
"""
Weighted Kernel Density Estimation (KDE) for Geospatial Data.

This script provides tools to calculate a weighted Gaussian KDE for both raster
(GeoTIFF) and vector (CSV/Excel) data. The primary application is to model the
influence of point sources (e.g., river estuaries with varying runoff) on a
spatial domain.

Main functionalities:
- A memory-efficient, chunked implementation of weighted Gaussian KDE.
- A function to apply the KDE to all valid pixels of a GeoTIFF raster file.
- A function to apply the KDE to a set of points from a CSV or Excel file.
"""

import warnings
from typing import Union
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial.distance import cdist
from tqdm import tqdm

def normalize_to_range(
    x: np.ndarray, 
    new_min: Union[int, float] = 0.0, 
    new_max: Union[int, float] = 1.0
) -> np.ndarray:
    """
    Normalizes a numpy array to a specified minimum and maximum range.
    NaN values in the input array are ignored in the calculation of min/max
    and will remain NaN in the output.

    Args:
        x (np.ndarray): The input numpy array to be normalized.
        new_min (Union[int, float], optional): The minimum value of the new range. Defaults to 0.0.
        new_max (Union[int, float], optional): The maximum value of the new range. Defaults to 1.0.

    Returns:
        np.ndarray: The normalized array.
    """
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if x_min == x_max:
        # Avoid division by zero if all values are the same
        return np.full_like(x, new_min, dtype=np.float64)
    return (x - x_min) * (new_max - new_min) / (x_max - x_min) + new_min

def gaussian_kde_weighted_chunked(
    points: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
    chunk_size: int = 50000
) -> np.ndarray:
    """
    Calculates weighted Gaussian Kernel Density Estimation in chunks to conserve memory.

    The weights associated with the centers are first normalized to the [0, 1] range.
    The final KDE result is also normalized to the [0, 1] range.

    Args:
        points (np.ndarray): An (N, 2) array of N points for which to calculate the density.
        centers (np.ndarray): An (M, 2) array of M kernel centers.
        weights (np.ndarray): A 1D array of M weights corresponding to each center.
        bandwidth (float): The bandwidth (h) of the Gaussian kernel. Must be positive.
        chunk_size (int, optional): The number of points to process in each batch. Defaults to 50000.

    Returns:
        np.ndarray: A 1D array of N calculated and normalized KDE values for each point.
    """
    # Input validation
    if not (len(points.shape) == 2 and points.shape[1] == 2):
        raise ValueError("points must be an Nx2 array.")
    if not (len(centers.shape) == 2 and centers.shape[1] == 2):
        raise ValueError("centers must be an Mx2 array.")
    if not (len(weights) == centers.shape[0]):
        raise ValueError("Length of weights must match the number of centers.")
    if not bandwidth > 0:
        raise ValueError("Bandwidth must be a positive number.")

    n_points = points.shape[0]
    result = np.zeros(n_points)
    
    # Normalize weights to ensure their scale does not excessively influence the result
    normalized_weights = normalize_to_range(weights)

    # Process points in chunks to manage memory usage
    for start in tqdm(range(0, n_points, chunk_size), desc="Calculating KDE"):
        end = min(start + chunk_size, n_points)
        batch_points = points[start:end]

        # Compute pairwise distances between batch points and all centers
        distances = cdist(batch_points, centers)  # Shape: (chunk_size, M)

        # Calculate Gaussian kernels
        with warnings.catch_warnings():
            # Ignore potential overflow/underflow warnings in np.exp,
            # as underflow to 0.0 is the expected behavior for large distances.
            warnings.simplefilter("ignore")
            kernels = np.exp(-0.5 * (distances / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

        # Apply weights to kernels and sum them up for each point
        batch_kde = kernels @ normalized_weights
        result[start:end] = batch_kde
    
    # Normalize the final result to the [0, 1] range for consistent interpretation
    return normalize_to_range(result)

def apply_kde_to_raster(
    input_tif_path: str,
    output_tif_path: str,
    centers_df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    weight_col: str,
    bandwidth: float,
    chunk_size: int = 50000
):
    """
    Applies weighted KDE to a raster file.

    The KDE is calculated for all pixels that are not NaN and have a value > 0.
    The result is saved as a new GeoTIFF file.

    Args:
        input_tif_path (str): Path to the input GeoTIFF file.
        output_tif_path (str): Path to save the output GeoTIFF file.
        centers_df (pd.DataFrame): DataFrame containing the kernel centers and their weights.
        lon_col (str): Name of the longitude column in centers_df.
        lat_col (str): Name of the latitude column in centers_df.
        weight_col (str): Name of the weight column in centers_df.
        bandwidth (float): The bandwidth for the KDE calculation.
        chunk_size (int, optional): The processing chunk size. Defaults to 50000.
    """
    try:
        with rasterio.open(input_tif_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            source_data = src.read(1)
            height, width = source_data.shape

            # Create a mask for valid data pixels (not NaN and > 0)
            valid_mask = (~np.isnan(source_data)) & (source_data > 0)
            rows, cols = np.where(valid_mask)

            # Convert pixel coordinates (row, col) to spatial coordinates (x, y)
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            points_to_process = np.column_stack((np.array(xs), np.array(ys)))

            # Extract center coordinates and weights from the DataFrame
            estuary_coords = centers_df[[lon_col, lat_col]].values
            weights = centers_df[weight_col].values

            print(f"Processing {len(points_to_process)} valid pixels from raster...")
            kde_values = gaussian_kde_weighted_chunked(
                points_to_process, estuary_coords, weights, bandwidth, chunk_size
            )

            # Prepare the output raster array
            kde_raster = np.full((height, width), np.nan, dtype=np.float32)
            kde_raster[rows, cols] = kde_values

            # Update raster profile for output
            profile.update(
                dtype=rasterio.float32,
                nodata=np.nan,
                compress='lzw'
            )

            # Write the result to a new GeoTIFF file
            with rasterio.open(output_tif_path, 'w', **profile) as dst:
                dst.write(kde_raster, 1)

            print(f"KDE calculation for raster complete. Result saved to: {output_tif_path}")

    except Exception as e:
        print(f"An error occurred during raster processing: {e}")

def apply_kde_to_vector(
    input_file_path: str,
    output_file_path: str,
    centers_df: pd.DataFrame,
    point_lon_col: str,
    point_lat_col: str,
    center_lon_col: str,
    center_lat_col: str,
    weight_col: str,
    bandwidth: float,
    chunk_size: int = 50000
):
    """
    Applies weighted KDE to a set of vector points from a file.

    Reads points from an Excel or CSV file, calculates the KDE value for each point,
    adds the result as a new column, and saves it to a new file.

    Args:
        input_file_path (str): Path to the input data file (.xlsx or .csv).
        output_file_path (str): Path to save the output file (.xlsx or .csv).
        centers_df (pd.DataFrame): DataFrame containing the kernel centers and their weights.
        point_lon_col (str): Column name for longitude in the input point file.
        point_lat_col (str): Column name for latitude in the input point file.
        center_lon_col (str): Column name for longitude in centers_df.
        center_lat_col (str): Column name for latitude in centers_df.
        weight_col (str): Column name for weights in centers_df.
        bandwidth (float): The bandwidth for the KDE calculation.
        chunk_size (int, optional): The processing chunk size. Defaults to 50000.
    """
    try:
        # Read input data
        if input_file_path.endswith('.xlsx'):
            df = pd.read_excel(input_file_path, index_col=0, header=0)
        elif input_file_path.endswith('.csv'):
            df = pd.read_csv(input_file_path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx or .csv")

        # Extract coordinates and weights
        points_to_process = df[[point_lon_col, point_lat_col]].values
        centers = centers_df[[center_lon_col, center_lat_col]].values
        weights = centers_df[weight_col].values

        print(f"Processing {len(points_to_process)} points from vector file...")
        kde_result = gaussian_kde_weighted_chunked(
            points_to_process, centers, weights, bandwidth, chunk_size
        )
        
        df['KDE'] = kde_result

        # Save results
        if output_file_path.endswith('.xlsx'):
            df.to_excel(output_file_path)
        elif output_file_path.endswith('.csv'):
            df.to_csv(output_file_path, index=False)
        else:
            raise ValueError("Unsupported output format. Please use .xlsx or .csv")
            
        print(f"KDE calculation for vector data complete. Result saved to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred during vector processing: {e}")


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # 1. Define common parameters for the analysis
    
    # DataFrame containing the centers (e.g., estuaries) and their weights (e.g., runoff)
    # In a real application, this could be loaded from a CSV file.
    estuaries_data = pd.DataFrame({
        'NAME': ['Humen', 'Jiaomen', 'Hongqimen', 'Hengmen', 'Modaomen', 'Jitimen', 'Hutiaomen', 'Yamen'],
        'LON': [113.627, 113.572, 113.551, 113.532, 113.410, 113.284, 113.133, 113.070],
        'LAT': [22.785, 22.735, 22.655, 22.580, 22.192, 22.134, 22.226, 22.308],
        'Annual_runoff': [603, 565, 209, 365, 923, 197, 202, 196]
    })
    
    # Scientific parameters
    # The bandwidth is a critical parameter and should be chosen based on the spatial scale of the process.
    # It is expressed in the same units as the coordinate system of the data (e.g., degrees).
    KDE_BANDWIDTH = 0.4
    PROCESSING_CHUNK_SIZE = 50000

    # 2. Example 1: Process a Raster (GeoTIFF) file
    print("--- Running Raster KDE Example ---")
    apply_kde_to_raster(
        input_tif_path=r"path/to/your/input_image.tif", # <-- IMPORTANT: Change this path
        output_tif_path=r"path/to/your/output_KDE_raster.tif", # <-- IMPORTANT: Change this path
        centers_df=estuaries_data,
        lon_col='LON',
        lat_col='LAT',
        weight_col='Annual_runoff',
        bandwidth=KDE_BANDWIDTH,
        chunk_size=PROCESSING_CHUNK_SIZE
    )
    print("\n" + "="*50 + "\n")

    # 3. Example 2: Process a Vector (Excel/CSV) file
    print("--- Running Vector KDE Example ---")
    apply_kde_to_vector(
        input_file_path=r"path/to/your/input_points.xlsx", # <-- IMPORTANT: Change this path
        output_file_path=r"path/to/your/output_KDE_points.xlsx", # <-- IMPORTANT: Change this path
        centers_df=estuaries_data,
        point_lon_col='X',         # Column name for longitude in the input points file
        point_lat_col='Y',         # Column name for latitude in the input points file
        center_lon_col='LON',      # Column name for longitude in the centers DataFrame
        center_lat_col='LAT',      # Column name for latitude in the centers DataFrame
        weight_col='Annual_runoff',# Column name for weights in the centers DataFrame
        bandwidth=KDE_BANDWIDTH,
        chunk_size=PROCESSING_CHUNK_SIZE
    )
