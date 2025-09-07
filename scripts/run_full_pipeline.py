#!/usr/bin/env python3
"""
Full Pipeline Runner for Hadapsar POI-VIIRS Analysis

This script runs the complete analysis pipeline from data preprocessing
to visualization generation.

Usage:
    python scripts/run_full_pipeline.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.foursquare_processor import FoursquareProcessor
from preprocessing.viirs_processor import VIIRSProcessor
from analysis.data_integration import DataIntegrator
from analysis.exploratory_analysis import ExploratoryAnalyzer
from visualization.interactive_maps import InteractiveMapVisualizer
from visualization.plotly_dashboard import PlotlyDashboard

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_preprocessing(data_dir: str) -> tuple:
    """
    Run data preprocessing pipeline.
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Tuple of (poi_output_file, viirs_output_file)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)
    
    # Initialize processors
    foursquare_processor = FoursquareProcessor(data_dir=data_dir)
    viirs_processor = VIIRSProcessor(data_dir=data_dir)
    
    # Process Foursquare POI data
    logger.info("Processing Foursquare POI data...")
    poi_output_file = foursquare_processor.process_foursquare_data()
    logger.info(f"POI processing complete: {poi_output_file}")
    
    # Process VIIRS data
    logger.info("Processing VIIRS nighttime lights data...")
    viirs_output_file, viirs_stats = viirs_processor.process_viirs_data(year=2023)
    logger.info(f"VIIRS processing complete: {viirs_output_file}")
    
    return poi_output_file, viirs_output_file

def run_integration(data_dir: str) -> str:
    """
    Run data integration pipeline.
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Path to integrated dataset
    """
    logger.info("=" * 60)
    logger.info("STEP 2: DATA INTEGRATION")
    logger.info("=" * 60)
    
    # Initialize integrator
    integrator = DataIntegrator(data_dir=data_dir)
    
    # Run integration pipeline
    logger.info("Integrating POI and VIIRS datasets...")
    integrated_gdf = integrator.integrate_datasets()
    
    # The integrated dataset is saved automatically
    integrated_file = integrator.processed_dir / "integrated_poi_viirs.geojson"
    logger.info(f"Integration complete: {integrated_file}")
    
    return str(integrated_file)

def run_analysis(data_dir: str, output_dir: str) -> dict:
    """
    Run exploratory analysis pipeline.
    
    Args:
        data_dir: Data directory path
        output_dir: Output directory path
        
    Returns:
        Analysis report dictionary
    """
    logger.info("=" * 60)
    logger.info("STEP 3: EXPLORATORY ANALYSIS")
    logger.info("=" * 60)
    
    # Initialize analyzer
    analyzer = ExploratoryAnalyzer(data_dir=data_dir, output_dir=output_dir)
    
    # Load integrated data
    logger.info("Loading integrated dataset...")
    integrated_gdf = analyzer.load_integrated_data()
    
    # Generate comprehensive analysis report
    logger.info("Generating comprehensive analysis report...")
    analysis_report = analyzer.generate_analysis_report(integrated_gdf)
    
    logger.info(f"Analysis complete. Report saved to: {analyzer.output_dir}/exploratory_analysis_report.json")
    
    return analysis_report

def run_visualization(data_dir: str, output_dir: str) -> dict:
    """
    Run visualization pipeline.
    
    Args:
        data_dir: Data directory path
        output_dir: Output directory path
        
    Returns:
        Dictionary of visualization file paths
    """
    logger.info("=" * 60)
    logger.info("STEP 4: VISUALIZATION")
    logger.info("=" * 60)
    
    # Initialize visualizers
    map_visualizer = InteractiveMapVisualizer(data_dir=data_dir, output_dir=output_dir)
    dashboard_creator = PlotlyDashboard(data_dir=data_dir, output_dir=output_dir)
    
    visualization_outputs = {}
    
    # Create interactive maps
    logger.info("Creating interactive maps...")
    try:
        comprehensive_map = map_visualizer.create_comprehensive_map()
        visualization_outputs['comprehensive_map'] = comprehensive_map
        logger.info(f"Comprehensive map created: {comprehensive_map}")
        
        category_maps = map_visualizer.create_category_specific_maps()
        visualization_outputs['category_maps'] = category_maps
        logger.info(f"Created {len(category_maps)} category-specific maps")
        
        temporal_map = map_visualizer.create_temporal_comparison_map()
        visualization_outputs['temporal_map'] = temporal_map
        logger.info(f"Temporal comparison map created: {temporal_map}")
        
    except Exception as e:
        logger.error(f"Error creating maps: {e}")
        visualization_outputs['maps_error'] = str(e)
    
    # Create interactive dashboards
    logger.info("Creating interactive dashboards...")
    try:
        dashboards = dashboard_creator.create_all_dashboards()
        visualization_outputs['dashboards'] = dashboards
        logger.info(f"Created {len(dashboards)} interactive dashboards")
        
    except Exception as e:
        logger.error(f"Error creating dashboards: {e}")
        visualization_outputs['dashboards_error'] = str(e)
    
    return visualization_outputs

def create_final_report(data_dir: str, output_dir: str, analysis_report: dict, 
                       visualization_outputs: dict) -> str:
    """
    Create final pipeline execution report.
    
    Args:
        data_dir: Data directory path
        output_dir: Output directory path
        analysis_report: Analysis results
        visualization_outputs: Visualization file paths
        
    Returns:
        Path to final report
    """
    logger.info("=" * 60)
    logger.info("CREATING FINAL REPORT")
    logger.info("=" * 60)
    
    import json
    from datetime import datetime
    
    # Compile final report
    final_report = {
        "pipeline_execution": {
            "timestamp": datetime.now().isoformat(),
            "data_directory": str(data_dir),
            "output_directory": str(output_dir),
            "status": "completed"
        },
        "data_summary": {
            "total_pois": analysis_report.get('descriptive_stats', {}).get('total_pois', 'N/A'),
            "categories": analysis_report.get('descriptive_stats', {}).get('unique_categories', 'N/A'),
            "luminosity_range": {
                "min": analysis_report.get('descriptive_stats', {}).get('luminosity_stats', {}).get('min', 'N/A'),
                "max": analysis_report.get('descriptive_stats', {}).get('luminosity_stats', {}).get('max', 'N/A'),
                "mean": analysis_report.get('descriptive_stats', {}).get('luminosity_stats', {}).get('mean', 'N/A')
            }
        },
        "analysis_results": {
            "correlation_analysis": bool(analysis_report.get('correlation_analysis')),
            "spatial_analysis": bool(analysis_report.get('spatial_analysis')),
            "cluster_analysis": bool(analysis_report.get('cluster_analysis')),
            "anomaly_analysis": bool(analysis_report.get('anomaly_analysis'))
        },
        "visualization_outputs": visualization_outputs,
        "key_findings": {
            "strongest_correlation": "See detailed analysis report",
            "cluster_quality": analysis_report.get('cluster_analysis', {}).get('cluster_quality', 'N/A'),
            "anomaly_rate": "See detailed analysis report",
            "spatial_pattern": analysis_report.get('spatial_analysis', {}).get('spatial_autocorrelation', 'N/A')
        },
        "recommendations": [
            "Review interactive maps for spatial patterns",
            "Examine dashboards for detailed analysis",
            "Investigate identified anomalies for planning insights",
            "Use findings for urban development decisions"
        ]
    }
    
    # Save final report
    report_path = Path(output_dir) / "final_pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"Final report saved: {report_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total POIs processed: {final_report['data_summary']['total_pois']}")
    logger.info(f"Categories analyzed: {final_report['data_summary']['categories']}")
    logger.info(f"Visualizations created: {len(visualization_outputs)}")
    logger.info(f"Final report: {report_path}")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return str(report_path)

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(
        description="Run the complete Hadapsar POI-VIIRS analysis pipeline"
    )
    parser.add_argument(
        "--data-dir", 
        default="data",
        help="Data directory path (default: data)"
    )
    parser.add_argument(
        "--output-dir", 
        default="results",
        help="Output directory path (default: results)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing steps (assumes data is already processed)"
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip integration step (assumes integrated data exists)"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis step"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization step"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    data_dir = str(Path(args.data_dir).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    
    logger.info("Starting Hadapsar POI-VIIRS Analysis Pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create directories if they don't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Preprocessing
        if not args.skip_preprocessing:
            poi_file, viirs_file = run_preprocessing(data_dir)
        else:
            logger.info("Skipping preprocessing step")
        
        # Step 2: Integration
        if not args.skip_integration:
            integrated_file = run_integration(data_dir)
        else:
            logger.info("Skipping integration step")
        
        # Step 3: Analysis
        if not args.skip_analysis:
            analysis_report = run_analysis(data_dir, output_dir)
        else:
            logger.info("Skipping analysis step")
            analysis_report = {}
        
        # Step 4: Visualization
        if not args.skip_visualization:
            visualization_outputs = run_visualization(data_dir, output_dir)
        else:
            logger.info("Skipping visualization step")
            visualization_outputs = {}
        
        # Step 5: Final Report
        final_report_path = create_final_report(data_dir, output_dir, 
                                              analysis_report, visualization_outputs)
        
        logger.info(f"🎉 Pipeline completed successfully!")
        logger.info(f"📊 Final report: {final_report_path}")
        logger.info(f"📁 All outputs saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())