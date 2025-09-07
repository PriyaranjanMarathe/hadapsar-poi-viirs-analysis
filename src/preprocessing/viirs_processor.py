"""
VIIRS Nighttime Lights Data Preprocessing Module

This module handles downloading, clipping, and preprocessing of VIIRS nighttime lights data
for the Hadapsar, Pune region analysis.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import box, Point
import requests
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VIIRSProcessor:
    """Processor for VIIRS nighttime lights data with focus on Hadapsar, Pune region."""
    
    # Hadapsar, Pune approximate bounding box (lat, lon)
    HADAPSAR_BOUNDS = {
        'lat_min': 18.4950,
        'lat_max': 18.5150,
        'lon_min': 73.9200,
        'lon_max': 73.9500
    }
    
    # VIIRS data URLs (example - users need to update with actual URLs)
    VIIRS_BASE_URL = "https://eogdata.mines.edu/nighttime_light/annual/v21/"
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the VIIRS processor.
        
        Args:
            data_dir: Directory to store processed data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create study area geometry
        self.study_area = self._create_study_area_polygon()
    
    def _create_study_area_polygon(self) -> gpd.GeoDataFrame:
        """Create a polygon for the study area (Hadapsar)."""
        bounds = self.HADAPSAR_BOUNDS
        geometry = box(bounds['lon_min'], bounds['lat_min'], 
                      bounds['lon_max'], bounds['lat_max'])
        
        gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs='EPSG:4326')
        return gdf
    
    def download_viirs_data(self, year: int = 2023, 
                           tile: str = "75N060E") -> Optional[str]:
        """
        Download VIIRS nighttime lights data for specified year and tile.
        
        Args:
            year: Year of data to download
            tile: VIIRS tile identifier covering the study area
            
        Returns:
            Path to downloaded TIFF file, or None if download fails
        """
        try:
            # Construct download URL
            filename = f"VNL_v21_npp_{year}_global_vcmslcfg_c202202151838.average_masked.tif"
            url = f"{self.VIIRS_BASE_URL}/{year}/{filename}"
            
            local_file = self.raw_dir / f"viirs_{year}_global.tif"
            
            # Check if file already exists
            if local_file.exists():
                logger.info(f"VIIRS data already exists: {local_file}")
                return str(local_file)
            
            logger.info(f"Downloading VIIRS data from {url}")
            
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(local_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify it's a valid TIFF file
                try:
                    with rasterio.open(local_file) as test:
                        pass
                    logger.info(f"Downloaded VIIRS data to {local_file}")
                    return str(local_file)
                except Exception as e:
                    logger.warning(f"Downloaded file is not a valid raster: {e}")
                    local_file.unlink(missing_ok=True)  # Delete invalid file
                    return self._create_sample_viirs_data(year)
            else:
                logger.warning(f"Failed to download VIIRS data. Status: {response.status_code}")
                return self._create_sample_viirs_data(year)
                
        except Exception as e:
            logger.error(f"Error downloading VIIRS data: {e}")
            logger.info("Creating sample VIIRS data instead")
            return self._create_sample_viirs_data(year)
    
    def _create_sample_viirs_data(self, year: int = 2023) -> str:
        """
        Create sample VIIRS-like raster data for testing.
        
        Args:
            year: Year for the sample data
            
        Returns:
            Path to sample TIFF file
        """
        bounds = self.HADAPSAR_BOUNDS
        
        # Create synthetic nighttime lights data
        width, height = 200, 200
        x = np.linspace(bounds['lon_min'], bounds['lon_max'], width)
        y = np.linspace(bounds['lat_min'], bounds['lat_max'], height)
        
        # Create synthetic data with some patterns
        xx, yy = np.meshgrid(x, y)
        
        # Simulate urban lighting patterns
        center_x, center_y = 73.9350, 18.5050  # Approximate center of Hadapsar
        distance_from_center = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        
        # Create radiance values that decrease with distance from center
        radiance = np.maximum(0, 50 - distance_from_center * 1000)
        
        # Add some random noise and hotspots
        np.random.seed(42)
        noise = np.random.normal(0, 5, radiance.shape)
        radiance = np.maximum(0, radiance + noise)
        
        # Add some random hotspots
        for _ in range(10):
            hx, hy = np.random.randint(0, width), np.random.randint(0, height)
            radiance[max(0, hy-5):min(height, hy+5), 
                    max(0, hx-5):min(width, hx+5)] += np.random.uniform(20, 80)
        
        # Create transform
        transform = rasterio.transform.from_bounds(
            bounds['lon_min'], bounds['lat_min'], 
            bounds['lon_max'], bounds['lat_max'], 
            width, height
        )
        
        # Save as GeoTIFF
        sample_file = self.raw_dir / f"viirs_{year}_sample.tif"
        
        with rasterio.open(
            sample_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=radiance.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(radiance, 1)
        
        logger.info(f"Created sample VIIRS data: {sample_file}")
        return str(sample_file)
    
    def clip_to_study_area(self, input_raster: str) -> str:
        """
        Clip VIIRS raster to study area boundaries.
        
        Args:
            input_raster: Path to input VIIRS raster
            
        Returns:
            Path to clipped raster
        """
        output_file = self.processed_dir / f"viirs_hadapsar_clipped.tif"
        
        with rasterio.open(input_raster) as src:
            # Get the geometry in the same CRS as the raster
            study_area_reproj = self.study_area.to_crs(src.crs)
            
            # Clip the raster
            out_image, out_transform = mask(
                src, 
                study_area_reproj.geometry, 
                crop=True,
                filled=False
            )
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Write clipped raster
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image)
        
        logger.info(f"Clipped VIIRS data to study area: {output_file}")
        return str(output_file)
    
    def calculate_statistics(self, raster_file: str) -> Dict:
        """
        Calculate statistics for the VIIRS raster.
        
        Args:
            raster_file: Path to VIIRS raster file
            
        Returns:
            Dictionary with statistics
        """
        with rasterio.open(raster_file) as src:
            data = src.read(1, masked=True)
            
            stats = {
                'count': np.sum(~data.mask),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'median': float(np.median(data)),
                'sum': float(np.sum(data))
            }
            
            # Calculate percentiles
            percentiles = [10, 25, 75, 90, 95, 99]
            for p in percentiles:
                stats[f'p{p}'] = float(np.percentile(data, p))
        
        logger.info(f"Calculated statistics for {raster_file}")
        return stats
    
    def create_luminosity_zones(self, raster_file: str) -> str:
        """
        Create zones based on luminosity levels.
        
        Args:
            raster_file: Path to VIIRS raster file
            
        Returns:
            Path to zone classification raster
        """
        output_file = self.processed_dir / "viirs_luminosity_zones.tif"
        
        with rasterio.open(raster_file) as src:
            data = src.read(1, masked=True)
            
            # Define luminosity thresholds
            zones = np.zeros_like(data, dtype=np.uint8)
            
            # Zone classification based on radiance values
            zones[data <= 1] = 1      # Very Low
            zones[(data > 1) & (data <= 5)] = 2    # Low
            zones[(data > 5) & (data <= 15)] = 3   # Medium
            zones[(data > 15) & (data <= 35)] = 4  # High
            zones[data > 35] = 5      # Very High
            
            # Copy metadata and update
            out_meta = src.meta.copy()
            out_meta.update(dtype='uint8', nodata=0)
            
            with rasterio.open(output_file, 'w', **out_meta) as dst:
                dst.write(zones, 1)
        
        logger.info(f"Created luminosity zones: {output_file}")
        return str(output_file)
    
    def extract_pixel_values_to_points(self, raster_file: str, 
                                      points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extract raster values to point locations.
        
        Args:
            raster_file: Path to VIIRS raster file
            points_gdf: GeoDataFrame with point locations
            
        Returns:
            GeoDataFrame with added luminosity values
        """
        with rasterio.open(raster_file) as src:
            # Convert points to raster CRS if needed
            if points_gdf.crs != src.crs:
                points_reproj = points_gdf.to_crs(src.crs)
            else:
                points_reproj = points_gdf.copy()
            
            # Extract coordinates
            coords = [(point.x, point.y) for point in points_reproj.geometry]
            
            # Sample raster values
            luminosity_values = [val[0] for val in src.sample(coords)]
            
            # Add to original GeoDataFrame
            result_gdf = points_gdf.copy()
            result_gdf['viirs_luminosity'] = luminosity_values
            
        logger.info(f"Extracted VIIRS values to {len(result_gdf)} points")
        return result_gdf
    
    def process_viirs_data(self, year: int = 2023, 
                          input_file: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Complete processing pipeline for VIIRS data.
        
        Args:
            year: Year of VIIRS data to process
            input_file: Path to input VIIRS file. If None, downloads data.
            
        Returns:
            Tuple of (processed_file_path, statistics_dict)
        """
        logger.info(f"Starting VIIRS data processing pipeline for year {year}")
        
        # Download or use existing file
        if input_file is None:
            input_file = self.download_viirs_data(year)
        
        if input_file is None:
            raise ValueError("Could not obtain VIIRS data")
        
        # Clip to study area
        clipped_file = self.clip_to_study_area(input_file)
        
        # Calculate statistics
        stats = self.calculate_statistics(clipped_file)
        
        # Create luminosity zones
        zones_file = self.create_luminosity_zones(clipped_file)
        
        # Save statistics to JSON
        import json
        stats_file = self.processed_dir / f"viirs_statistics_{year}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"VIIRS processing complete. Files saved:")
        logger.info(f"  Clipped raster: {clipped_file}")
        logger.info(f"  Luminosity zones: {zones_file}")
        logger.info(f"  Statistics: {stats_file}")
        
        # Print summary
        self._print_summary_stats(stats)
        
        return clipped_file, stats
    
    def _print_summary_stats(self, stats: Dict):
        """Print summary statistics of processed VIIRS data."""
        logger.info("\n=== VIIRS Processing Summary ===")
        logger.info(f"Total pixels: {stats['count']:,}")
        logger.info(f"Radiance range: {stats['min']:.2f} to {stats['max']:.2f}")
        logger.info(f"Mean radiance: {stats['mean']:.2f}")
        logger.info(f"Median radiance: {stats['median']:.2f}")
        logger.info(f"Total radiance sum: {stats['sum']:,.2f}")
        logger.info(f"95th percentile: {stats['p95']:.2f}")


if __name__ == "__main__":
    # Example usage
    processor = VIIRSProcessor()
    output_file, stats = processor.process_viirs_data(2023)
    print(f"Processed VIIRS data saved to: {output_file}")