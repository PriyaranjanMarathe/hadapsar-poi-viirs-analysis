"""
Plotly Dashboard Module

This module creates interactive dashboards using Plotly for analyzing POI-VIIRS integration results.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py
from pathlib import Path
import json
import logging
from typing import Optional, Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlotlyDashboard:
    """Class for creating interactive Plotly dashboards for POI-VIIRS analysis."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        """
        Initialize the dashboard creator.
        
        Args:
            data_dir: Directory containing processed data
            output_dir: Directory to save visualizations
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "dashboards").mkdir(exist_ok=True)
        
        # Color schemes
        self.category_colors = {
            'Commercial': '#FF6B6B',
            'Essential Services': '#4ECDC4',
            'Education': '#45B7D1',
            'Recreation': '#96CEB4',
            'Other': '#FECA57'
        }
    
    def load_data(self) -> Tuple[gpd.GeoDataFrame, Optional[Dict]]:
        """
        Load integrated data and statistics.
        
        Returns:
            Tuple of (poi_gdf, statistics_dict)
        """
        # Load integrated POI data
        poi_file = self.processed_dir / "integrated_poi_viirs.geojson"
        if not poi_file.exists():
            raise FileNotFoundError(f"Integrated POI data not found: {poi_file}")
        
        poi_gdf = gpd.read_file(poi_file)
        
        # Load statistics if available
        stats_file = self.processed_dir / "integration_statistics.json"
        stats = None
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        
        logger.info(f"Loaded {len(poi_gdf)} POIs for dashboard")
        return poi_gdf, stats
    
    def create_overview_dashboard(self, poi_gdf: gpd.GeoDataFrame, stats: Optional[Dict] = None) -> str:
        """
        Create comprehensive overview dashboard.
        
        Args:
            poi_gdf: GeoDataFrame with integrated data
            stats: Statistics dictionary
            
        Returns:
            Path to saved HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('POI Distribution', 'Luminosity Distribution', 'Category vs Luminosity',
                          'Spatial Distribution', 'Cluster Analysis', 'Temporal Patterns',
                          'Correlation Matrix', 'Anomaly Analysis', 'Summary Statistics'),
            specs=[[{"type": "pie"}, {"type": "histogram"}, {"type": "box"}],
                   [{"type": "scattermapbox"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "pie"}, {"type": "table"}]]
        )
        
        # 1. POI Distribution (Pie Chart)
        category_counts = poi_gdf['category_group'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                name="POI Distribution",
                marker_colors=[self.category_colors.get(cat, '#888888') for cat in category_counts.index]
            ),
            row=1, col=1
        )
        
        # 2. Luminosity Distribution (Histogram)
        if 'viirs_luminosity' in poi_gdf.columns:
            fig.add_trace(
                go.Histogram(
                    x=poi_gdf['viirs_luminosity'],
                    name="Luminosity Distribution",
                    nbinsx=30,
                    marker_color='skyblue'
                ),
                row=1, col=2
            )
        
        # 3. Category vs Luminosity (Box Plot)
        if 'viirs_luminosity' in poi_gdf.columns:
            for i, category in enumerate(poi_gdf['category_group'].unique()):
                category_data = poi_gdf[poi_gdf['category_group'] == category]
                fig.add_trace(
                    go.Box(
                        y=category_data['viirs_luminosity'],
                        name=category,
                        marker_color=self.category_colors.get(category, '#888888'),
                        showlegend=False
                    ),
                    row=1, col=3
                )
        
        # 4. Spatial Distribution (Scatter Map)
        fig.add_trace(
            go.Scattermapbox(
                lat=poi_gdf.geometry.y,
                lon=poi_gdf.geometry.x,
                mode='markers',
                marker=dict(
                    size=8,
                    color=[self.category_colors.get(cat, '#888888') for cat in poi_gdf['category_group']]
                ),
                text=poi_gdf['category_group'],
                name="POI Locations"
            ),
            row=2, col=1
        )
        
        # 5. Cluster Analysis (Bar Chart)
        if 'cluster' in poi_gdf.columns:
            cluster_counts = poi_gdf['cluster'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=[f"Cluster {i}" for i in cluster_counts.index],
                    y=cluster_counts.values,
                    name="Cluster Distribution",
                    marker_color='lightcoral'
                ),
                row=2, col=2
            )
        
        # 6. Temporal Patterns (Placeholder - using luminosity over categories)
        if 'viirs_luminosity' in poi_gdf.columns:
            avg_luminosity = poi_gdf.groupby('category_group')['viirs_luminosity'].mean()
            fig.add_trace(
                go.Scatter(
                    x=avg_luminosity.index,
                    y=avg_luminosity.values,
                    mode='lines+markers',
                    name="Avg Luminosity by Category",
                    line=dict(color='orange')
                ),
                row=2, col=3
            )
        
        # 7. Correlation Matrix (simplified)
        if 'viirs_luminosity' in poi_gdf.columns:
            # Create dummy variables for categories
            category_dummies = pd.get_dummies(poi_gdf['category_group'])
            corr_data = pd.concat([category_dummies, poi_gdf['viirs_luminosity']], axis=1).corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.columns,
                    colorscale='RdBu',
                    zmid=0,
                    name="Correlation Matrix"
                ),
                row=3, col=1
            )
        
        # 8. Anomaly Analysis (Pie Chart)
        if 'anomaly_type' in poi_gdf.columns:
            anomaly_counts = poi_gdf['anomaly_type'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=anomaly_counts.index,
                    values=anomaly_counts.values,
                    name="Anomaly Distribution"
                ),
                row=3, col=2
            )
        
        # 9. Summary Statistics (Table)
        if stats:
            summary_data = [
                ['Total POIs', stats.get('total_pois', 'N/A')],
                ['Categories', stats.get('categories', 'N/A')],
                ['Mean Luminosity', f"{stats.get('luminosity', {}).get('mean', 0):.2f}"],
                ['Max Luminosity', f"{stats.get('luminosity', {}).get('max', 0):.2f}"],
                ['Study Area (km²)', 'Hadapsar, Pune']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value']),
                    cells=dict(values=[[row[0] for row in summary_data],
                                     [row[1] for row in summary_data]]),
                    name="Summary Statistics"
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Hadapsar POI-VIIRS Analysis Dashboard",
            title_x=0.5,
            showlegend=False,
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=18.5050, lon=73.9350),
                zoom=13
            )
        )
        
        # Save dashboard
        dashboard_file = self.output_dir / "dashboards" / "overview_dashboard.html"
        py.plot(fig, filename=str(dashboard_file), auto_open=False)
        
        logger.info(f"Overview dashboard saved to {dashboard_file}")
        return str(dashboard_file)
    
    def create_spatial_analysis_dashboard(self, poi_gdf: gpd.GeoDataFrame) -> str:
        """
        Create spatial analysis focused dashboard.
        
        Args:
            poi_gdf: GeoDataFrame with integrated data
            
        Returns:
            Path to saved HTML dashboard
        """
        # Create main spatial plot
        fig = go.Figure()
        
        # Add POI scatter plot
        for category in poi_gdf['category_group'].unique():
            category_data = poi_gdf[poi_gdf['category_group'] == category]
            
            fig.add_trace(go.Scattermapbox(
                lat=category_data.geometry.y,
                lon=category_data.geometry.x,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.category_colors.get(category, '#888888'),
                    opacity=0.7
                ),
                text=category_data.apply(lambda x: f"{x['name']}<br>Category: {x['category_group']}<br>Luminosity: {x.get('viirs_luminosity', 0):.2f}", axis=1),
                name=category,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout for map
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=18.5050, lon=73.9350),
                zoom=14
            ),
            height=800,
            title="Spatial Distribution of POIs in Hadapsar",
            title_x=0.5
        )
        
        # Save spatial dashboard
        dashboard_file = self.output_dir / "dashboards" / "spatial_analysis_dashboard.html"
        py.plot(fig, filename=str(dashboard_file), auto_open=False)
        
        logger.info(f"Spatial analysis dashboard saved to {dashboard_file}")
        return str(dashboard_file)
    
    def create_correlation_dashboard(self, poi_gdf: gpd.GeoDataFrame) -> str:
        """
        Create correlation analysis dashboard.
        
        Args:
            poi_gdf: GeoDataFrame with integrated data
            
        Returns:
            Path to saved HTML dashboard
        """
        if 'viirs_luminosity' not in poi_gdf.columns:
            logger.warning("No luminosity data available for correlation analysis")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Luminosity by Category', 'Scatter Plot: Coordinates vs Luminosity',
                          'Luminosity Distribution by Category', 'Category Luminosity Statistics'),
            specs=[[{"type": "box"}, {"type": "scatter"}],
                   [{"type": "violin"}, {"type": "bar"}]]
        )
        
        # 1. Box plot of luminosity by category
        for category in poi_gdf['category_group'].unique():
            category_data = poi_gdf[poi_gdf['category_group'] == category]
            fig.add_trace(
                go.Box(
                    y=category_data['viirs_luminosity'],
                    name=category,
                    marker_color=self.category_colors.get(category, '#888888')
                ),
                row=1, col=1
            )
        
        # 2. Scatter plot of coordinates vs luminosity
        fig.add_trace(
            go.Scatter(
                x=poi_gdf.geometry.x,
                y=poi_gdf['viirs_luminosity'],
                mode='markers',
                marker=dict(
                    color=poi_gdf['viirs_luminosity'],
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title="Luminosity")
                ),
                text=poi_gdf['category_group'],
                name="Longitude vs Luminosity"
            ),
            row=1, col=2
        )
        
        # 3. Violin plot of luminosity distribution by category
        for category in poi_gdf['category_group'].unique():
            category_data = poi_gdf[poi_gdf['category_group'] == category]
            fig.add_trace(
                go.Violin(
                    y=category_data['viirs_luminosity'],
                    name=category,
                    box_visible=True,
                    line_color=self.category_colors.get(category, '#888888'),
                    fillcolor=self.category_colors.get(category, '#888888'),
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # 4. Bar chart of mean luminosity by category
        mean_luminosity = poi_gdf.groupby('category_group')['viirs_luminosity'].mean()
        fig.add_trace(
            go.Bar(
                x=mean_luminosity.index,
                y=mean_luminosity.values,
                marker_color=[self.category_colors.get(cat, '#888888') for cat in mean_luminosity.index],
                name="Mean Luminosity"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="POI-VIIRS Correlation Analysis",
            title_x=0.5,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_file = self.output_dir / "dashboards" / "correlation_dashboard.html"
        py.plot(fig, filename=str(dashboard_file), auto_open=False)
        
        logger.info(f"Correlation dashboard saved to {dashboard_file}")
        return str(dashboard_file)
    
    def create_cluster_dashboard(self, poi_gdf: gpd.GeoDataFrame) -> str:
        """
        Create cluster analysis dashboard.
        
        Args:
            poi_gdf: GeoDataFrame with integrated data
            
        Returns:
            Path to saved HTML dashboard
        """
        if 'cluster' not in poi_gdf.columns:
            logger.warning("No cluster data available")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Spatial Distribution', 'Cluster Characteristics',
                          'Cluster Sizes', 'Cluster Luminosity Comparison'),
            specs=[[{"type": "scattermapbox"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Define cluster colors
        cluster_colors = px.colors.qualitative.Set1
        
        # 1. Spatial distribution of clusters
        for cluster_id in sorted(poi_gdf['cluster'].unique()):
            cluster_data = poi_gdf[poi_gdf['cluster'] == cluster_id]
            color = cluster_colors[cluster_id % len(cluster_colors)]
            
            fig.add_trace(
                go.Scattermapbox(
                    lat=cluster_data.geometry.y,
                    lon=cluster_data.geometry.x,
                    mode='markers',
                    marker=dict(size=10, color=color),
                    text=[f"Cluster {cluster_id}<br>Category: {cat}<br>Luminosity: {lum:.2f}" 
                          for cat, lum in zip(cluster_data['category_group'], 
                                            cluster_data.get('viirs_luminosity', [0]*len(cluster_data)))],
                    name=f"Cluster {cluster_id}"
                ),
                row=1, col=1
            )
        
        # 2. Cluster characteristics (PCA-like visualization)
        if 'viirs_luminosity' in poi_gdf.columns:
            fig.add_trace(
                go.Scatter(
                    x=poi_gdf.geometry.x,
                    y=poi_gdf['viirs_luminosity'],
                    mode='markers',
                    marker=dict(
                        color=poi_gdf['cluster'],
                        colorscale='tab10',
                        size=8,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Cluster {c}" for c in poi_gdf['cluster']],
                    name="Cluster Characteristics"
                ),
                row=1, col=2
            )
        
        # 3. Cluster sizes
        cluster_counts = poi_gdf['cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {i}" for i in cluster_counts.index],
                y=cluster_counts.values,
                marker_color=[cluster_colors[i % len(cluster_colors)] for i in cluster_counts.index],
                name="Cluster Sizes"
            ),
            row=2, col=1
        )
        
        # 4. Luminosity comparison by cluster
        if 'viirs_luminosity' in poi_gdf.columns:
            for cluster_id in sorted(poi_gdf['cluster'].unique()):
                cluster_data = poi_gdf[poi_gdf['cluster'] == cluster_id]
                fig.add_trace(
                    go.Box(
                        y=cluster_data['viirs_luminosity'],
                        name=f"Cluster {cluster_id}",
                        marker_color=cluster_colors[cluster_id % len(cluster_colors)]
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Cluster Analysis Dashboard",
            title_x=0.5,
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=18.5050, lon=73.9350),
                zoom=13
            )
        )
        
        # Save dashboard
        dashboard_file = self.output_dir / "dashboards" / "cluster_dashboard.html"
        py.plot(fig, filename=str(dashboard_file), auto_open=False)
        
        logger.info(f"Cluster dashboard saved to {dashboard_file}")
        return str(dashboard_file)
    
    def create_all_dashboards(self) -> Dict[str, str]:
        """
        Create all dashboard types.
        
        Returns:
            Dictionary mapping dashboard type to file path
        """
        logger.info("Creating all interactive dashboards")
        
        # Load data
        poi_gdf, stats = self.load_data()
        
        dashboards = {}
        
        # Create overview dashboard
        try:
            dashboards['overview'] = self.create_overview_dashboard(poi_gdf, stats)
        except Exception as e:
            logger.error(f"Error creating overview dashboard: {e}")
        
        # Create spatial analysis dashboard
        try:
            dashboards['spatial'] = self.create_spatial_analysis_dashboard(poi_gdf)
        except Exception as e:
            logger.error(f"Error creating spatial dashboard: {e}")
        
        # Create correlation dashboard
        try:
            dashboards['correlation'] = self.create_correlation_dashboard(poi_gdf)
        except Exception as e:
            logger.error(f"Error creating correlation dashboard: {e}")
        
        # Create cluster dashboard
        try:
            dashboards['cluster'] = self.create_cluster_dashboard(poi_gdf)
        except Exception as e:
            logger.error(f"Error creating cluster dashboard: {e}")
        
        logger.info(f"Created {len(dashboards)} dashboards")
        return dashboards


if __name__ == "__main__":
    # Example usage
    dashboard_creator = PlotlyDashboard()
    
    # Create all dashboards
    dashboards = dashboard_creator.create_all_dashboards()
    
    print("Created dashboards:")
    for dashboard_type, file_path in dashboards.items():
        print(f"  {dashboard_type}: {file_path}")
    
    print("Interactive dashboard creation complete!")