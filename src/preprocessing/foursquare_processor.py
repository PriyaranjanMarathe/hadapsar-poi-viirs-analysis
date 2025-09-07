"""
Foursquare POI Data Preprocessing Module

This module handles downloading, filtering, and preprocessing of Foursquare POI data
for the Hadapsar, Pune region analysis.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import boto3
import s3fs
import pyarrow.parquet as pq
from typing import Optional, Tuple, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoursquareProcessor:
    """Processor for Foursquare POI data with focus on Hadapsar, Pune region."""
    
    # Hadapsar, Pune approximate bounding box (lat, lon)
    HADAPSAR_BOUNDS = {
        'lat_min': 18.4950,
        'lat_max': 18.5150,
        'lon_min': 73.9200,
        'lon_max': 73.9500
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Foursquare processor.
        
        Args:
            data_dir: Directory to store processed data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_foursquare_data(self, 
                                region: str = "india",
                                category_filter: Optional[List[str]] = None) -> str:
        """
        Download Foursquare POI data from S3.
        
        Args:
            region: Geographic region to download
            category_filter: List of category IDs to filter
            
        Returns:
            Path to downloaded parquet file
        """
        try:
            # Note: This is a placeholder for the actual Foursquare S3 path
            # Users need to replace with actual S3 bucket and path
            s3_path = f"s3://foursquare-oss-places/release/dt=2023-12-01/places/country=IN/region={region}/"
            
            fs = s3fs.S3FileSystem(anon=True)
            
            # List available files
            files = fs.ls(s3_path)
            if not files:
                logger.warning(f"No files found at {s3_path}")
                return None
            
            # Download the first available parquet file
            remote_file = files[0]
            local_file = self.raw_dir / f"foursquare_{region}.parquet"
            
            logger.info(f"Downloading {remote_file} to {local_file}")
            fs.get(remote_file, str(local_file))
            
            return str(local_file)
            
        except Exception as e:
            logger.error(f"Error downloading Foursquare data: {e}")
            logger.info("Using sample data generation instead")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> str:
        """
        Create sample POI data for testing purposes.
        
        Returns:
            Path to sample parquet file
        """
        sample_data = {
            'fsq_id': [f"poi_{i:04d}" for i in range(100)],
            'name': [f"Sample POI {i}" for i in range(100)],
            'latitude': [18.5000 + (i % 20) * 0.001 for i in range(100)],
            'longitude': [73.9300 + (i % 20) * 0.001 for i in range(100)],
            'category_id': [f"cat_{i % 10:02d}" for i in range(100)],
            'category_name': [
                ['Restaurant', 'Shop', 'Bank', 'Hospital', 'School', 
                 'Park', 'Gas Station', 'Pharmacy', 'ATM', 'Hotel'][i % 10]
                for i in range(100)
            ],
            'address': [f"Sample Address {i}, Hadapsar, Pune" for i in range(100)],
            'region': ['Maharashtra'] * 100,
            'country': ['IN'] * 100
        }
        
        df = pd.DataFrame(sample_data)
        sample_file = self.raw_dir / "foursquare_sample.parquet"
        df.to_parquet(sample_file, index=False)
        
        logger.info(f"Created sample data: {sample_file}")
        return str(sample_file)
    
    def filter_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter POIs to Hadapsar region based on coordinates.
        
        Args:
            df: Input dataframe with POI data
            
        Returns:
            Filtered dataframe
        """
        bounds = self.HADAPSAR_BOUNDS
        
        filtered_df = df[
            (df['latitude'] >= bounds['lat_min']) &
            (df['latitude'] <= bounds['lat_max']) &
            (df['longitude'] >= bounds['lon_min']) &
            (df['longitude'] <= bounds['lon_max'])
        ]
        
        logger.info(f"Filtered {len(df)} POIs to {len(filtered_df)} in Hadapsar region")
        return filtered_df
    
    def create_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Convert pandas DataFrame to GeoDataFrame.
        
        Args:
            df: Input dataframe with latitude/longitude columns
            
        Returns:
            GeoDataFrame with Point geometries
        """
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        logger.info(f"Created GeoDataFrame with {len(gdf)} POIs")
        return gdf
    
    def categorize_pois(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add category groupings for analysis.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with additional category columns
        """
        # Define category mappings
        commercial_categories = ['Restaurant', 'Shop', 'Hotel', 'Gas Station']
        essential_categories = ['Hospital', 'Pharmacy', 'Bank', 'ATM']
        education_categories = ['School']
        recreation_categories = ['Park']
        
        def get_category_group(category):
            if category in commercial_categories:
                return 'Commercial'
            elif category in essential_categories:
                return 'Essential Services'
            elif category in education_categories:
                return 'Education'
            elif category in recreation_categories:
                return 'Recreation'
            else:
                return 'Other'
        
        gdf['category_group'] = gdf['category_name'].apply(get_category_group)
        
        logger.info(f"Added category groupings")
        return gdf
    
    def process_foursquare_data(self, input_file: Optional[str] = None) -> str:
        """
        Complete processing pipeline for Foursquare data.
        
        Args:
            input_file: Path to input parquet file. If None, downloads data.
            
        Returns:
            Path to processed GeoDataFrame file
        """
        logger.info("Starting Foursquare data processing pipeline")
        
        # Download or use existing file
        if input_file is None:
            input_file = self.download_foursquare_data()
        
        if input_file is None:
            raise ValueError("Could not obtain Foursquare data")
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)
        
        # Filter to region
        df_filtered = self.filter_by_region(df)
        
        # Create GeoDataFrame
        gdf = self.create_geodataframe(df_filtered)
        
        # Add category groupings
        gdf = self.categorize_pois(gdf)
        
        # Save processed data
        output_file = self.processed_dir / "hadapsar_pois_processed.geojson"
        gdf.to_file(output_file, driver='GeoJSON')
        
        # Also save as parquet for faster loading
        parquet_file = self.processed_dir / "hadapsar_pois_processed.parquet"
        gdf.drop(columns=['geometry']).to_parquet(parquet_file)
        
        logger.info(f"Processed data saved to {output_file} and {parquet_file}")
        
        # Print summary statistics
        self._print_summary_stats(gdf)
        
        return str(output_file)
    
    def _print_summary_stats(self, gdf: gpd.GeoDataFrame):
        """Print summary statistics of processed data."""
        logger.info("\n=== Foursquare POI Processing Summary ===")
        logger.info(f"Total POIs: {len(gdf)}")
        logger.info(f"Unique categories: {gdf['category_name'].nunique()}")
        logger.info("\nPOIs by category group:")
        category_counts = gdf['category_group'].value_counts()
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count}")
        
        logger.info(f"\nBounding box:")
        bounds = gdf.bounds
        logger.info(f"  Lat: {bounds.miny.min():.6f} to {bounds.maxy.max():.6f}")
        logger.info(f"  Lon: {bounds.minx.min():.6f} to {bounds.maxx.max():.6f}")


if __name__ == "__main__":
    # Example usage
    processor = FoursquareProcessor()
    output_file = processor.process_foursquare_data()
    print(f"Processed Foursquare data saved to: {output_file}")