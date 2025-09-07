"""
Exploratory Data Analysis Module

This module provides comprehensive exploratory analysis functions for the integrated
POI-VIIRS dataset, including correlation analysis, spatial patterns, and statistical summaries.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExploratoryAnalyzer:
    """Class for performing exploratory data analysis on integrated POI-VIIRS data."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        """
        Initialize the exploratory analyzer.
        
        Args:
            data_dir: Directory containing processed data
            output_dir: Directory to save analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.data_dir / "processed"
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
    
    def load_integrated_data(self, file_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load integrated POI-VIIRS dataset.
        
        Args:
            file_path: Path to integrated data file
            
        Returns:
            GeoDataFrame with integrated data
        """
        if file_path is None:
            file_path = self.processed_dir / "integrated_poi_viirs.geojson"
        
        if not Path(file_path).exists():
            logger.error(f"Integrated data file not found: {file_path}")
            raise FileNotFoundError(f"Integrated data file not found: {file_path}")
        
        gdf = gpd.read_file(file_path)
        logger.info(f"Loaded integrated data: {len(gdf)} records")
        return gdf
    
    def basic_descriptive_stats(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Calculate basic descriptive statistics.
        
        Args:
            gdf: GeoDataFrame with integrated data
            
        Returns:
            Dictionary with descriptive statistics
        """
        stats_dict = {}
        
        # Overall statistics
        stats_dict['total_pois'] = len(gdf)
        stats_dict['unique_categories'] = gdf['category_group'].nunique()
        
        # Luminosity statistics
        if 'viirs_luminosity' in gdf.columns:
            lum_stats = gdf['viirs_luminosity'].describe()
            stats_dict['luminosity_stats'] = lum_stats.to_dict()
        
        # Category distribution
        category_counts = gdf['category_group'].value_counts()
        stats_dict['category_distribution'] = category_counts.to_dict()
        
        # Spatial statistics
        bounds = gdf.total_bounds
        stats_dict['spatial_extent'] = {
            'min_longitude': float(bounds[0]),
            'min_latitude': float(bounds[1]),
            'max_longitude': float(bounds[2]),
            'max_latitude': float(bounds[3]),
            'width_km': self._calculate_distance(bounds[0], bounds[1], bounds[2], bounds[1]),
            'height_km': self._calculate_distance(bounds[0], bounds[1], bounds[0], bounds[3])
        }
        
        logger.info("Calculated basic descriptive statistics")
        return stats_dict
    
    def _calculate_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate approximate distance in km between two points."""
        # Simplified distance calculation
        lat_diff = abs(lat2 - lat1) * 111  # 1 degree latitude ≈ 111 km
        lon_diff = abs(lon2 - lon1) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def poi_luminosity_correlation_analysis(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Analyze correlation between POI categories and luminosity.
        
        Args:
            gdf: GeoDataFrame with integrated data
            
        Returns:
            Dictionary with correlation results
        """
        results = {}
        
        # Overall correlation
        if 'viirs_luminosity' in gdf.columns:
            # Create dummy variables for categories
            category_dummies = pd.get_dummies(gdf['category_group'])
            
            # Combine with luminosity data
            analysis_df = pd.concat([category_dummies, gdf['viirs_luminosity']], axis=1)
            
            # Calculate correlation matrix
            correlation_matrix = analysis_df.corr()
            
            # Extract correlations with luminosity
            luminosity_correlations = correlation_matrix['viirs_luminosity'].drop('viirs_luminosity')
            results['category_luminosity_correlations'] = luminosity_correlations.to_dict()
            
            # Statistical significance tests
            results['correlation_tests'] = {}
            for category in category_dummies.columns:
                category_data = gdf[gdf['category_group'] == category]['viirs_luminosity']
                other_data = gdf[gdf['category_group'] != category]['viirs_luminosity']
                
                # Mann-Whitney U test (non-parametric)
                try:
                    statistic, p_value = stats.mannwhitneyu(category_data, other_data, 
                                                          alternative='two-sided')
                    results['correlation_tests'][category] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'mean_luminosity': float(category_data.mean()),
                        'median_luminosity': float(category_data.median())
                    }
                except Exception as e:
                    logger.warning(f"Could not perform test for {category}: {e}")
        
        logger.info("Completed correlation analysis")
        return results
    
    def spatial_autocorrelation_analysis(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Analyze spatial autocorrelation patterns.
        
        Args:
            gdf: GeoDataFrame with integrated data
            
        Returns:
            Dictionary with spatial autocorrelation results
        """
        results = {}
        
        if 'viirs_luminosity' in gdf.columns and len(gdf) > 10:
            # Extract coordinates
            coords = np.array([[p.x, p.y] for p in gdf.geometry])
            luminosity = gdf['viirs_luminosity'].values
            
            # Calculate distance matrix (limit to reasonable sample size)
            sample_size = min(500, len(gdf))
            if len(gdf) > sample_size:
                sample_idx = np.random.choice(len(gdf), sample_size, replace=False)
                coords_sample = coords[sample_idx]
                luminosity_sample = luminosity[sample_idx]
            else:
                coords_sample = coords
                luminosity_sample = luminosity
            
            # Calculate pairwise distances
            distances = pdist(coords_sample)
            distance_matrix = squareform(distances)
            
            # Moran's I approximation
            try:
                n = len(luminosity_sample)
                mean_lum = np.mean(luminosity_sample)
                
                # Weight matrix (inverse distance, with threshold)
                max_distance = np.percentile(distances, 80)  # Use 80th percentile as threshold
                weights = np.where(distance_matrix > 0, 1 / distance_matrix, 0)
                weights[distance_matrix > max_distance] = 0
                
                # Normalize weights
                row_sums = np.sum(weights, axis=1)
                weights = np.divide(weights, row_sums[:, np.newaxis], 
                                  out=np.zeros_like(weights), where=row_sums[:, np.newaxis]!=0)
                
                # Calculate Moran's I
                numerator = 0
                denominator = 0
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            numerator += weights[i, j] * (luminosity_sample[i] - mean_lum) * (luminosity_sample[j] - mean_lum)
                
                for i in range(n):
                    denominator += (luminosity_sample[i] - mean_lum) ** 2
                
                W = np.sum(weights)
                if W > 0 and denominator > 0:
                    morans_i = (n / W) * (numerator / denominator)
                    results['morans_i'] = float(morans_i)
                    results['spatial_autocorrelation'] = 'positive' if morans_i > 0 else 'negative'
                    results['autocorrelation_strength'] = abs(morans_i)
                
            except Exception as e:
                logger.warning(f"Could not calculate Moran's I: {e}")
                results['morans_i'] = None
        
        logger.info("Completed spatial autocorrelation analysis")
        return results
    
    def cluster_analysis(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Analyze cluster characteristics.
        
        Args:
            gdf: GeoDataFrame with integrated data
            
        Returns:
            Dictionary with cluster analysis results
        """
        results = {}
        
        if 'cluster' in gdf.columns:
            # Cluster summary statistics
            cluster_stats = gdf.groupby('cluster').agg({
                'viirs_luminosity': ['count', 'mean', 'std', 'median'],
                'category_group': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
            }).round(3)
            
            results['cluster_summary'] = cluster_stats.to_dict()
            
            # Silhouette analysis
            if 'viirs_luminosity' in gdf.columns and len(gdf) > 10:
                try:
                    # Prepare features for silhouette analysis
                    features = []
                    features.append(gdf['viirs_luminosity'].values.reshape(-1, 1))
                    
                    # Add coordinates
                    coords = np.array([[p.x, p.y] for p in gdf.geometry])
                    features.append(coords)
                    
                    X = np.hstack(features)
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(X_scaled, gdf['cluster'])
                    results['silhouette_score'] = float(silhouette_avg)
                    results['cluster_quality'] = 'good' if silhouette_avg > 0.5 else 'moderate' if silhouette_avg > 0.3 else 'poor'
                    
                except Exception as e:
                    logger.warning(f"Could not calculate silhouette score: {e}")
        
        logger.info("Completed cluster analysis")
        return results
    
    def anomaly_analysis(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Analyze anomalous patterns in the data.
        
        Args:
            gdf: GeoDataFrame with integrated data
            
        Returns:
            Dictionary with anomaly analysis results
        """
        results = {}
        
        if 'anomaly_type' in gdf.columns:
            # Anomaly distribution
            anomaly_counts = gdf['anomaly_type'].value_counts()
            results['anomaly_distribution'] = anomaly_counts.to_dict()
            
            # Anomaly characteristics
            anomaly_stats = {}
            for anomaly_type in gdf['anomaly_type'].unique():
                anomaly_data = gdf[gdf['anomaly_type'] == anomaly_type]
                
                if 'viirs_luminosity' in gdf.columns:
                    anomaly_stats[anomaly_type] = {
                        'count': len(anomaly_data),
                        'mean_luminosity': float(anomaly_data['viirs_luminosity'].mean()),
                        'median_luminosity': float(anomaly_data['viirs_luminosity'].median()),
                        'dominant_category': anomaly_data['category_group'].mode().iloc[0] if len(anomaly_data) > 0 else None
                    }
            
            results['anomaly_characteristics'] = anomaly_stats
        
        logger.info("Completed anomaly analysis")
        return results
    
    def create_summary_plots(self, gdf: gpd.GeoDataFrame):
        """
        Create summary plots for exploratory analysis.
        
        Args:
            gdf: GeoDataFrame with integrated data
        """
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hadapsar POI-VIIRS Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Category distribution
        if 'category_group' in gdf.columns:
            category_counts = gdf['category_group'].value_counts()
            axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('POI Category Distribution')
        
        # 2. Luminosity distribution
        if 'viirs_luminosity' in gdf.columns:
            axes[0, 1].hist(gdf['viirs_luminosity'], bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].set_xlabel('VIIRS Luminosity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Luminosity Distribution')
        
        # 3. Category vs Luminosity boxplot
        if 'category_group' in gdf.columns and 'viirs_luminosity' in gdf.columns:
            sns.boxplot(data=gdf, x='category_group', y='viirs_luminosity', ax=axes[0, 2])
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].set_title('Luminosity by Category')
        
        # 4. Spatial distribution
        axes[1, 0].scatter(gdf.geometry.x, gdf.geometry.y, alpha=0.6, s=10)
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        axes[1, 0].set_title('Spatial Distribution of POIs')
        
        # 5. Cluster distribution
        if 'cluster' in gdf.columns:
            cluster_counts = gdf['cluster'].value_counts().sort_index()
            axes[1, 1].bar(range(len(cluster_counts)), cluster_counts.values)
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Cluster Distribution')
            axes[1, 1].set_xticks(range(len(cluster_counts)))
        
        # 6. Anomaly distribution
        if 'anomaly_type' in gdf.columns:
            anomaly_counts = gdf['anomaly_type'].value_counts()
            axes[1, 2].pie(anomaly_counts.values, labels=anomaly_counts.index, autopct='%1.1f%%')
            axes[1, 2].set_title('Anomaly Type Distribution')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "plots" / "exploratory_analysis_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary plots saved to {plot_file}")
    
    def generate_analysis_report(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Args:
            gdf: GeoDataFrame with integrated data
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive exploratory analysis")
        
        report = {}
        
        # Basic descriptive statistics
        report['descriptive_stats'] = self.basic_descriptive_stats(gdf)
        
        # Correlation analysis
        report['correlation_analysis'] = self.poi_luminosity_correlation_analysis(gdf)
        
        # Spatial autocorrelation
        report['spatial_analysis'] = self.spatial_autocorrelation_analysis(gdf)
        
        # Cluster analysis
        report['cluster_analysis'] = self.cluster_analysis(gdf)
        
        # Anomaly analysis
        report['anomaly_analysis'] = self.anomaly_analysis(gdf)
        
        # Create summary plots
        self.create_summary_plots(gdf)
        
        # Save comprehensive report
        import json
        report_file = self.output_dir / "exploratory_analysis_report.json"
        
        # Convert any non-JSON serializable keys to strings
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_report = make_json_serializable(report)
        
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive analysis report saved to {report_file}")
        
        # Print key findings
        self._print_key_findings(report)
        
        return report
    
    def _print_key_findings(self, report: Dict):
        """Print key findings from the analysis."""
        logger.info("\n=== KEY FINDINGS FROM EXPLORATORY ANALYSIS ===")
        
        # Basic stats
        if 'descriptive_stats' in report:
            stats = report['descriptive_stats']
            logger.info(f"Total POIs analyzed: {stats['total_pois']}")
            logger.info(f"Categories present: {stats['unique_categories']}")
            
            if 'luminosity_stats' in stats:
                logger.info(f"Luminosity range: {stats['luminosity_stats']['min']:.2f} to {stats['luminosity_stats']['max']:.2f}")
                logger.info(f"Mean luminosity: {stats['luminosity_stats']['mean']:.2f}")
        
        # Correlation findings
        if 'correlation_analysis' in report and 'category_luminosity_correlations' in report['correlation_analysis']:
            correlations = report['correlation_analysis']['category_luminosity_correlations']
            logger.info("\nCategory-Luminosity Correlations:")
            for category, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                logger.info(f"  {category}: {corr:.3f}")
        
        # Spatial patterns
        if 'spatial_analysis' in report and 'morans_i' in report['spatial_analysis']:
            morans_i = report['spatial_analysis']['morans_i']
            if morans_i is not None:
                logger.info(f"\nSpatial autocorrelation (Moran's I): {morans_i:.3f}")
                logger.info(f"Spatial pattern: {report['spatial_analysis'].get('spatial_autocorrelation', 'Unknown')}")
        
        # Cluster quality
        if 'cluster_analysis' in report and 'silhouette_score' in report['cluster_analysis']:
            silhouette = report['cluster_analysis']['silhouette_score']
            quality = report['cluster_analysis']['cluster_quality']
            logger.info(f"\nCluster quality: {quality} (Silhouette score: {silhouette:.3f})")
        
        # Anomalies
        if 'anomaly_analysis' in report and 'anomaly_distribution' in report['anomaly_analysis']:
            anomalies = report['anomaly_analysis']['anomaly_distribution']
            logger.info("\nAnomaly distribution:")
            for anomaly_type, count in anomalies.items():
                logger.info(f"  {anomaly_type}: {count}")


if __name__ == "__main__":
    # Example usage
    analyzer = ExploratoryAnalyzer()
    
    # Load data and generate report
    integrated_data = analyzer.load_integrated_data()
    report = analyzer.generate_analysis_report(integrated_data)
    
    print("Exploratory analysis complete. Check the results directory for outputs.")