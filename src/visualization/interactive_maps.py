"""
Interactive Map Visualization Module

This module creates interactive maps using Folium for visualizing POI-VIIRS integration results
in Hadapsar, Pune.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium import plugins
import rasterio
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from branca.colormap import LinearColormap
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveMapVisualizer:
    """Class for creating interactive maps of POI-VIIRS analysis results."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        """
        Initialize the interactive map visualizer.
        
        Args:
            data_dir: Directory containing processed data
            output_dir: Directory to save visualizations
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "maps").mkdir(exist_ok=True)
        
        # Hadapsar center coordinates for map centering
        self.center_lat = 18.5050
        self.center_lon = 73.9350
        
        # Category color mapping
        self.category_colors = {
            'Commercial': '#FF6B6B',
            'Essential Services': '#4ECDC4',
            'Education': '#45B7D1',
            'Recreation': '#96CEB4',
            'Other': '#FECA57'
        }
    
    def load_data(self) -> Tuple[gpd.GeoDataFrame, Optional[np.ndarray]]:
        """
        Load integrated POI data and VIIRS raster.
        
        Returns:
            Tuple of (poi_gdf, viirs_array)
        """
        # Load POI data
        poi_file = self.processed_dir / "integrated_poi_viirs.geojson"
        if not poi_file.exists():
            raise FileNotFoundError(f"Integrated POI data not found: {poi_file}")
        
        poi_gdf = gpd.read_file(poi_file)
        logger.info(f"Loaded {len(poi_gdf)} POIs for visualization")
        
        # Load VIIRS raster (optional)
        viirs_file = self.processed_dir / "viirs_hadapsar_clipped.tif"
        viirs_array = None
        
        if viirs_file.exists():
            with rasterio.open(viirs_file) as src:
                viirs_array = src.read(1, masked=True)
            logger.info("Loaded VIIRS raster for visualization")
        
        return poi_gdf, viirs_array
    
    def create_base_map(self, zoom_start: int = 14) -> folium.Map:
        """
        Create base map centered on Hadapsar.
        
        Args:
            zoom_start: Initial zoom level
            
        Returns:
            Folium map object
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles=None
        )
        
        # Add different tile layers
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        return m
    
    def add_poi_markers(self, m: folium.Map, poi_gdf: gpd.GeoDataFrame,
                       color_by: str = 'category_group') -> folium.Map:
        """
        Add POI markers to the map.
        
        Args:
            m: Folium map object
            poi_gdf: GeoDataFrame with POI data
            color_by: Column to use for coloring markers
            
        Returns:
            Updated map object
        """
        # Create marker cluster
        marker_cluster = plugins.MarkerCluster(name='POI Markers').add_to(m)
        
        for idx, poi in poi_gdf.iterrows():
            # Determine marker color
            if color_by == 'category_group' and poi['category_group'] in self.category_colors:
                color = self.category_colors[poi['category_group']]
            elif color_by == 'viirs_luminosity' and 'viirs_luminosity' in poi_gdf.columns:
                # Color by luminosity (blue to red)
                lum_value = poi['viirs_luminosity']
                max_lum = poi_gdf['viirs_luminosity'].max()
                if max_lum > 0:
                    intensity = min(lum_value / max_lum, 1.0)
                    color = f'#{int(255 * intensity):02x}{int(255 * (1-intensity)):02x}00'
                else:
                    color = '#888888'
            else:
                color = '#888888'
            
            # Create popup content
            popup_content = self._create_poi_popup_content(poi)
            
            # Add marker
            folium.CircleMarker(
                location=[poi.geometry.y, poi.geometry.x],
                radius=6,
                popup=folium.Popup(popup_content, max_width=300),
                color='white',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(marker_cluster)
        
        logger.info(f"Added {len(poi_gdf)} POI markers to map")
        return m
    
    def _create_poi_popup_content(self, poi: pd.Series) -> str:
        """Create HTML content for POI popup."""
        content = f"""
        <div style='font-family: Arial, sans-serif;'>
            <h4 style='margin: 0 0 10px 0; color: #333;'>{poi.get('name', 'Unknown POI')}</h4>
            <p><strong>Category:</strong> {poi.get('category_group', 'Unknown')}</p>
            <p><strong>Specific Type:</strong> {poi.get('category_name', 'Unknown')}</p>
        """
        
        if 'viirs_luminosity' in poi.index:
            content += f"<p><strong>Nighttime Luminosity:</strong> {poi['viirs_luminosity']:.2f}</p>"
        
        if 'cluster' in poi.index:
            content += f"<p><strong>Cluster:</strong> {poi['cluster']}</p>"
        
        if 'anomaly_type' in poi.index:
            content += f"<p><strong>Pattern:</strong> {poi['anomaly_type']}</p>"
        
        content += "</div>"
        return content
    
    def add_luminosity_heatmap(self, m: folium.Map, poi_gdf: gpd.GeoDataFrame) -> folium.Map:
        """
        Add luminosity heatmap layer.
        
        Args:
            m: Folium map object
            poi_gdf: GeoDataFrame with POI and luminosity data
            
        Returns:
            Updated map object
        """
        if 'viirs_luminosity' not in poi_gdf.columns:
            logger.warning("No luminosity data available for heatmap")
            return m
        
        # Prepare heatmap data
        heat_data = []
        for idx, poi in poi_gdf.iterrows():
            if poi['viirs_luminosity'] > 0:  # Only include areas with luminosity
                heat_data.append([
                    poi.geometry.y, 
                    poi.geometry.x, 
                    float(poi['viirs_luminosity'])
                ])
        
        if heat_data:
            # Add heatmap layer
            heatmap = plugins.HeatMap(
                heat_data,
                name='Luminosity Heatmap',
                min_opacity=0.2,
                max_zoom=18,
                radius=20,
                blur=15,
                gradient={0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
            )
            heatmap.add_to(m)
            
            logger.info(f"Added luminosity heatmap with {len(heat_data)} points")
        
        return m
    
    def add_cluster_visualization(self, m: folium.Map, poi_gdf: gpd.GeoDataFrame) -> folium.Map:
        """
        Add cluster visualization to the map.
        
        Args:
            m: Folium map object
            poi_gdf: GeoDataFrame with cluster assignments
            
        Returns:
            Updated map object
        """
        if 'cluster' not in poi_gdf.columns:
            logger.warning("No cluster data available")
            return m
        
        # Define cluster colors
        cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                         '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
        
        # Create feature groups for each cluster
        for cluster_id in sorted(poi_gdf['cluster'].unique()):
            cluster_data = poi_gdf[poi_gdf['cluster'] == cluster_id]
            color = cluster_colors[cluster_id % len(cluster_colors)]
            
            cluster_group = folium.FeatureGroup(name=f'Cluster {cluster_id}')
            
            for idx, poi in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[poi.geometry.y, poi.geometry.x],
                    radius=8,
                    popup=f"Cluster {cluster_id}<br>Luminosity: {poi.get('viirs_luminosity', 0):.2f}",
                    color='white',
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(cluster_group)
            
            cluster_group.add_to(m)
        
        logger.info(f"Added cluster visualization for {poi_gdf['cluster'].nunique()} clusters")
        return m
    
    def add_anomaly_visualization(self, m: folium.Map, poi_gdf: gpd.GeoDataFrame) -> folium.Map:
        """
        Add anomaly visualization to the map.
        
        Args:
            m: Folium map object
            poi_gdf: GeoDataFrame with anomaly classifications
            
        Returns:
            Updated map object
        """
        if 'anomaly_type' not in poi_gdf.columns:
            logger.warning("No anomaly data available")
            return m
        
        # Define anomaly colors and symbols
        anomaly_config = {
            'Normal': {'color': '#28a745', 'icon': 'ok'},
            'High Light, Non-Commercial': {'color': '#ffc107', 'icon': 'warning-sign'},
            'Low Light, Commercial': {'color': '#dc3545', 'icon': 'exclamation-sign'}
        }
        
        # Create feature group for anomalies
        for anomaly_type, config in anomaly_config.items():
            anomaly_data = poi_gdf[poi_gdf['anomaly_type'] == anomaly_type]
            
            if len(anomaly_data) > 0:
                anomaly_group = folium.FeatureGroup(name=f'{anomaly_type} ({len(anomaly_data)})')
                
                for idx, poi in anomaly_data.iterrows():
                    popup_content = self._create_poi_popup_content(poi)
                    
                    if anomaly_type == 'Normal':
                        # Use circle markers for normal points
                        folium.CircleMarker(
                            location=[poi.geometry.y, poi.geometry.x],
                            radius=5,
                            popup=folium.Popup(popup_content, max_width=300),
                            color=config['color'],
                            fillOpacity=0.6
                        ).add_to(anomaly_group)
                    else:
                        # Use icon markers for anomalies
                        folium.Marker(
                            location=[poi.geometry.y, poi.geometry.x],
                            popup=folium.Popup(popup_content, max_width=300),
                            icon=folium.Icon(color='red' if 'High Light' in anomaly_type else 'orange', 
                                           icon=config['icon'])
                        ).add_to(anomaly_group)
                
                anomaly_group.add_to(m)
        
        logger.info("Added anomaly visualization")
        return m
    
    def create_comprehensive_map(self) -> str:
        """
        Create a comprehensive interactive map with all visualization layers.
        
        Returns:
            Path to saved HTML map
        """
        logger.info("Creating comprehensive interactive map")
        
        # Load data
        poi_gdf, viirs_array = self.load_data()
        
        # Create base map
        m = self.create_base_map()
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 40px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:16px; font-weight:bold; padding: 10px">
        Hadapsar POI-VIIRS Analysis - Interactive Map
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add different visualization layers
        self.add_poi_markers(m, poi_gdf, color_by='category_group')
        self.add_luminosity_heatmap(m, poi_gdf)
        
        if 'cluster' in poi_gdf.columns:
            self.add_cluster_visualization(m, poi_gdf)
        
        if 'anomaly_type' in poi_gdf.columns:
            self.add_anomaly_visualization(m, poi_gdf)
        
        # Add legend
        self._add_legend(m, poi_gdf)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add measurement tool
        plugins.MeasureControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save map
        map_file = self.output_dir / "maps" / "hadapsar_comprehensive_map.html"
        m.save(str(map_file))
        
        logger.info(f"Comprehensive map saved to {map_file}")
        return str(map_file)
    
    def _add_legend(self, m: folium.Map, poi_gdf: gpd.GeoDataFrame):
        """Add legend to the map."""
        # Category legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4 style="margin-top:0">POI Categories</h4>
        '''
        
        for category, color in self.category_colors.items():
            if category in poi_gdf['category_group'].values:
                count = len(poi_gdf[poi_gdf['category_group'] == category])
                legend_html += f'''
                <p><i class="fa fa-circle" style="color:{color}"></i> {category} ({count})</p>
                '''
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_category_specific_maps(self) -> List[str]:
        """
        Create separate maps for each POI category.
        
        Returns:
            List of paths to saved HTML maps
        """
        logger.info("Creating category-specific maps")
        
        # Load data
        poi_gdf, _ = self.load_data()
        
        map_files = []
        
        for category in poi_gdf['category_group'].unique():
            category_data = poi_gdf[poi_gdf['category_group'] == category]
            
            # Create map for this category
            m = self.create_base_map()
            
            # Add title
            title_html = f'''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 350px; height: 40px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:16px; font-weight:bold; padding: 10px">
            Hadapsar {category} POIs
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Add markers for this category
            for idx, poi in category_data.iterrows():
                popup_content = self._create_poi_popup_content(poi)
                
                folium.CircleMarker(
                    location=[poi.geometry.y, poi.geometry.x],
                    radius=8,
                    popup=folium.Popup(popup_content, max_width=300),
                    color='white',
                    weight=1,
                    fillColor=self.category_colors[category],
                    fillOpacity=0.8
                ).add_to(m)
            
            # Add statistics box
            stats_html = f'''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 200px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <h4 style="margin-top:0">{category} Statistics</h4>
            <p>Total POIs: {len(category_data)}</p>
            '''
            
            if 'viirs_luminosity' in category_data.columns:
                avg_lum = category_data['viirs_luminosity'].mean()
                stats_html += f'<p>Avg Luminosity: {avg_lum:.2f}</p>'
            
            stats_html += '</div>'
            m.get_root().html.add_child(folium.Element(stats_html))
            
            # Save map
            safe_category = category.replace(' ', '_').replace('/', '_')
            map_file = self.output_dir / "maps" / f"hadapsar_{safe_category.lower()}_map.html"
            m.save(str(map_file))
            map_files.append(str(map_file))
            
            logger.info(f"Created map for {category}: {map_file}")
        
        return map_files
    
    def create_temporal_comparison_map(self, years: List[int] = [2022, 2023]) -> str:
        """
        Create a map comparing temporal changes (placeholder for future implementation).
        
        Args:
            years: List of years to compare
            
        Returns:
            Path to saved HTML map
        """
        logger.info("Creating temporal comparison map (using current data as demo)")
        
        # Load current data
        poi_gdf, _ = self.load_data()
        
        # Create map
        m = self.create_base_map()
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 40px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:16px; font-weight:bold; padding: 10px">
        Hadapsar Temporal Analysis (Demo)
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add current POIs
        self.add_poi_markers(m, poi_gdf, color_by='category_group')
        self.add_luminosity_heatmap(m, poi_gdf)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        map_file = self.output_dir / "maps" / "hadapsar_temporal_comparison.html"
        m.save(str(map_file))
        
        logger.info(f"Temporal comparison map saved to {map_file}")
        return str(map_file)


if __name__ == "__main__":
    # Example usage
    visualizer = InteractiveMapVisualizer()
    
    # Create comprehensive map
    comprehensive_map = visualizer.create_comprehensive_map()
    print(f"Comprehensive map created: {comprehensive_map}")
    
    # Create category-specific maps
    category_maps = visualizer.create_category_specific_maps()
    print(f"Created {len(category_maps)} category-specific maps")
    
    print("Interactive map visualization complete!")