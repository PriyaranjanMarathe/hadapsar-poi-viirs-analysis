"""
Data Integration Module

This module handles the spatial integration of Foursquare POI data with VIIRS nighttime lights data
for urban analysis in Hadapsar, Pune.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, Polygon
from rasterio.features import rasterize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIntegrator:
    """Class for integrating POI and VIIRS data for spatial analysis."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data integrator.
        
        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        
    def load_poi_data(self, file_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load processed POI data.
        
        Args:
            file_path: Path to POI GeoJSON file. If None, uses default path.
            
        Returns:
            GeoDataFrame with POI data
        """
        if file_path is None:
            file_path = self.processed_dir / "hadapsar_pois_processed.geojson"
        
        if not Path(file_path).exists():
            logger.error(f"POI file not found: {file_path}")
            raise FileNotFoundError(f"POI file not found: {file_path}")
        
        gdf = gpd.read_file(file_path)
        logger.info(f"Loaded {len(gdf)} POIs from {file_path}")
        return gdf
    
    def load_viirs_data(self, file_path: Optional[str] = None) -> Tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
        """
        Load processed VIIRS raster data.
        
        Args:
            file_path: Path to VIIRS TIFF file. If None, uses default path.
            
        Returns:
            Tuple of (data_array, transform, crs)
        """
        if file_path is None:
            file_path = self.processed_dir / "viirs_hadapsar_clipped.tif"
        
        if not Path(file_path).exists():
            logger.error(f"VIIRS file not found: {file_path}")
            raise FileNotFoundError(f"VIIRS file not found: {file_path}")
        
        with rasterio.open(file_path) as src:
            data = src.read(1, masked=True)
            transform = src.transform
            crs = src.crs
        
        logger.info(f"Loaded VIIRS data from {file_path}")
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"  CRS: {crs}")
        return data, transform, crs
    
    def extract_luminosity_to_pois(self, poi_gdf: gpd.GeoDataFrame,
                                  viirs_file: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Extract VIIRS luminosity values to POI locations.
        
        Args:
            poi_gdf: GeoDataFrame with POI data
            viirs_file: Path to VIIRS raster file
            
        Returns:
            GeoDataFrame with added luminosity values
        """
        if viirs_file is None:
            viirs_file = self.processed_dir / "viirs_hadapsar_clipped.tif"
        
        with rasterio.open(viirs_file) as src:
            # Ensure POI data is in same CRS as raster
            if poi_gdf.crs != src.crs:
                poi_reproj = poi_gdf.to_crs(src.crs)
            else:
                poi_reproj = poi_gdf.copy()
            
            # Extract coordinates
            coords = [(point.x, point.y) for point in poi_reproj.geometry]
            
            # Sample raster values at POI locations
            luminosity_values = []
            for coord in coords:
                try:
                    value = next(src.sample([coord]))[0]
                    # Handle nodata values
                    if np.isnan(value) or value == src.nodata:
                        value = 0
                    luminosity_values.append(float(value))
                except Exception as e:
                    logger.warning(f"Could not sample at {coord}: {e}")
                    luminosity_values.append(0)
            
            # Add luminosity values to original GeoDataFrame
            result_gdf = poi_gdf.copy()
            result_gdf['viirs_luminosity'] = luminosity_values
        
        logger.info(f"Extracted VIIRS luminosity to {len(result_gdf)} POIs")
        return result_gdf
    
    def create_poi_density_raster(self, poi_gdf: gpd.GeoDataFrame, 
                                 template_raster: str,
                                 category_column: str = 'category_group') -> Dict[str, np.ndarray]:
        """
        Create raster layers showing POI density by category.
        
        Args:
            poi_gdf: GeoDataFrame with POI data
            template_raster: Path to template raster for grid alignment
            category_column: Column name for POI categories
            
        Returns:
            Dictionary of category -> density raster arrays
        """
        with rasterio.open(template_raster) as template:
            transform = template.transform
            shape = (template.height, template.width)
            crs = template.crs
        
        # Ensure POIs are in same CRS
        if poi_gdf.crs != crs:
            poi_reproj = poi_gdf.to_crs(crs)
        else:
            poi_reproj = poi_gdf.copy()
        
        density_rasters = {}
        categories = poi_reproj[category_column].unique()
        
        for category in categories:
            category_pois = poi_reproj[poi_reproj[category_column] == category]
            
            # Create point geometries with value 1 for rasterization
            geometries = [(geom, 1) for geom in category_pois.geometry]
            
            # Rasterize POI locations
            if geometries:
                poi_raster = rasterize(
                    geometries,
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint16
                )
                density_rasters[category] = poi_raster
            else:
                density_rasters[category] = np.zeros(shape, dtype=np.uint16)
        
        logger.info(f"Created density rasters for {len(categories)} POI categories")
        return density_rasters
    
    def calculate_spatial_statistics(self, integrated_gdf: gpd.GeoDataFrame) -> Dict:
        """
        Calculate spatial statistics for integrated dataset.
        
        Args:
            integrated_gdf: GeoDataFrame with POI and luminosity data
            
        Returns:
            Dictionary with statistical measures
        """
        stats = {}
        
        # Basic statistics
        stats['total_pois'] = len(integrated_gdf)
        stats['categories'] = integrated_gdf['category_group'].nunique()
        
        # Luminosity statistics
        lum_col = 'viirs_luminosity'
        if lum_col in integrated_gdf.columns:
            stats['luminosity'] = {
                'min': float(integrated_gdf[lum_col].min()),
                'max': float(integrated_gdf[lum_col].max()),
                'mean': float(integrated_gdf[lum_col].mean()),
                'median': float(integrated_gdf[lum_col].median()),
                'std': float(integrated_gdf[lum_col].std())
            }
            
            # Category-wise luminosity statistics
            stats['category_luminosity'] = {}
            for category in integrated_gdf['category_group'].unique():
                cat_data = integrated_gdf[integrated_gdf['category_group'] == category]
                stats['category_luminosity'][category] = {
                    'count': len(cat_data),
                    'mean_luminosity': float(cat_data[lum_col].mean()),
                    'median_luminosity': float(cat_data[lum_col].median())
                }
        
        # Spatial extent
        bounds = integrated_gdf.total_bounds
        stats['spatial_extent'] = {
            'minx': float(bounds[0]),
            'miny': float(bounds[1]),
            'maxx': float(bounds[2]),
            'maxy': float(bounds[3])
        }
        
        logger.info("Calculated spatial statistics")
        return stats
    
    def identify_poi_luminosity_clusters(self, integrated_gdf: gpd.GeoDataFrame,
                                       n_clusters: int = 5) -> gpd.GeoDataFrame:
        """
        Perform clustering based on POI density and luminosity.
        
        Args:
            integrated_gdf: GeoDataFrame with integrated data
            n_clusters: Number of clusters to create
            
        Returns:
            GeoDataFrame with cluster assignments
        """
        # Prepare features for clustering
        features = []
        
        # Use luminosity as primary feature
        if 'viirs_luminosity' in integrated_gdf.columns:
            features.append(integrated_gdf['viirs_luminosity'].values.reshape(-1, 1))
        
        # Add spatial coordinates
        coords = np.array([[p.x, p.y] for p in integrated_gdf.geometry])
        features.append(coords)
        
        # Combine features
        X = np.hstack(features)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster assignments
        result_gdf = integrated_gdf.copy()
        result_gdf['cluster'] = clusters
        
        logger.info(f"Created {n_clusters} clusters for {len(result_gdf)} POIs")
        
        # Log cluster summary
        cluster_summary = result_gdf.groupby('cluster').agg({
            'viirs_luminosity': ['count', 'mean'],
            'category_group': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
        }).round(2)
        
        logger.info("Cluster summary:")
        logger.info(cluster_summary)
        
        return result_gdf
    
    def find_anomalous_areas(self, integrated_gdf: gpd.GeoDataFrame,
                           luminosity_threshold: float = None) -> gpd.GeoDataFrame:
        """
        Identify areas with unusual POI-luminosity relationships.
        
        Args:
            integrated_gdf: GeoDataFrame with integrated data
            luminosity_threshold: Threshold for high/low luminosity classification
            
        Returns:
            GeoDataFrame with anomaly flags
        """
        if luminosity_threshold is None:
            luminosity_threshold = integrated_gdf['viirs_luminosity'].median()
        
        result_gdf = integrated_gdf.copy()
        
        # Classify areas
        high_light = result_gdf['viirs_luminosity'] > luminosity_threshold
        
        # Count POIs in neighborhood (simplified - could use buffer analysis)
        result_gdf['high_luminosity'] = high_light
        result_gdf['anomaly_type'] = 'Normal'
        
        # High light, few commercial POIs (potential residential/industrial)
        commercial_mask = result_gdf['category_group'] == 'Commercial'
        result_gdf.loc[high_light & ~commercial_mask, 'anomaly_type'] = 'High Light, Non-Commercial'
        
        # Low light, many commercial POIs (potential underlit commercial areas)
        result_gdf.loc[~high_light & commercial_mask, 'anomaly_type'] = 'Low Light, Commercial'
        
        anomaly_counts = result_gdf['anomaly_type'].value_counts()
        logger.info("Anomaly detection results:")
        for anomaly_type, count in anomaly_counts.items():
            logger.info(f"  {anomaly_type}: {count}")
        
        return result_gdf
    
    def integrate_datasets(self, poi_file: Optional[str] = None,
                          viirs_file: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Complete integration pipeline for POI and VIIRS data.
        
        Args:
            poi_file: Path to POI data file
            viirs_file: Path to VIIRS raster file
            
        Returns:
            GeoDataFrame with integrated data
        """
        logger.info("Starting data integration pipeline")
        
        # Load datasets
        poi_gdf = self.load_poi_data(poi_file)
        
        # Extract luminosity values to POIs
        integrated_gdf = self.extract_luminosity_to_pois(poi_gdf, viirs_file)
        
        # Perform clustering analysis
        clustered_gdf = self.identify_poi_luminosity_clusters(integrated_gdf)
        
        # Identify anomalous areas
        final_gdf = self.find_anomalous_areas(clustered_gdf)
        
        # Calculate statistics
        stats = self.calculate_spatial_statistics(final_gdf)
        
        # Save integrated dataset
        output_file = self.processed_dir / "integrated_poi_viirs.geojson"
        final_gdf.to_file(output_file, driver='GeoJSON')
        
        # Save statistics
        import json
        stats_file = self.processed_dir / "integration_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Integration complete. Files saved:")
        logger.info(f"  Integrated data: {output_file}")
        logger.info(f"  Statistics: {stats_file}")
        
        # Print summary
        self._print_integration_summary(final_gdf, stats)
        
        return final_gdf
    
    def _print_integration_summary(self, gdf: gpd.GeoDataFrame, stats: Dict):
        """Print summary of integration results."""
        logger.info("\n=== Data Integration Summary ===")
        logger.info(f"Total POIs with luminosity data: {len(gdf)}")
        logger.info(f"Categories: {stats['categories']}")
        logger.info(f"Luminosity range: {stats['luminosity']['min']:.2f} to {stats['luminosity']['max']:.2f}")
        logger.info(f"Mean luminosity: {stats['luminosity']['mean']:.2f}")
        
        logger.info("\nCluster distribution:")
        cluster_counts = gdf['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            logger.info(f"  Cluster {cluster}: {count} POIs")
        
        logger.info("\nAnomaly distribution:")
        anomaly_counts = gdf['anomaly_type'].value_counts()
        for anomaly_type, count in anomaly_counts.items():
            logger.info(f"  {anomaly_type}: {count}")


if __name__ == "__main__":
    # Example usage
    integrator = DataIntegrator()
    integrated_data = integrator.integrate_datasets()
    print(f"Integration complete. {len(integrated_data)} POIs processed.")