# hadapsar-poi-viirs-analysis

# README: Integrating Foursquare POI Data and VIIRS Nighttime Lights for Urban Analysis in Hadapsar, Pune

## Project Overview

This project explores the integration of **Foursquare Open Source Places** data and **VIIRS bias-corrected nighttime lights** data to analyze urban development, economic activity, and resource allocation in **Hadapsar, Pune**. By combining these datasets, we aim to uncover insights into urban growth patterns, infrastructure gaps, and sustainability challenges.

---

## Objectives

1. **Urban Growth Analysis**: Evaluate patterns of urbanization by correlating nighttime light intensity with the density and types of Points of Interest (POIs).
2. **Economic Insights**: Identify economic hotspots and underserved areas by comparing POI categories with nighttime illumination.
3. **Infrastructure Gaps**: Highlight regions with high nighttime lights but low POI presence (or vice versa) to pinpoint areas needing infrastructure or development.
4. **Scalability**: Develop a replicable methodology that can be applied to other regions globally.

---

## Key Research Questions

1. How do POI distributions correlate with nighttime light intensities in Hadapsar?
2. Can changes in lighting patterns over time predict urbanization trends in the area?
3. Are there disparities between economic activity (as inferred from POIs) and energy use (as inferred from lighting)?
4. How might this analysis guide urban planners in addressing growth or sustainability challenges?

---

## Data Sources

1. **Foursquare Open Source Places Dataset**:

   - Description: A global database of over 100 million POIs, including attributes like name, category, and geocoordinates.
   - Format: Parquet files hosted on Amazon S3.
   - License: Apache 2.0.

2. **VIIRS Nighttime Lights Data**:

   - Description: Bias-corrected satellite data measuring nighttime illumination, corrected for cloud cover and atmospheric conditions.
   - Format: GeoTIFF or raster data.

3. **Auxiliary Data (Optional)**:
   - Census data or urban planning maps for validation.
   - Local GIS layers for additional context.

---

## Methodology

### 1. Data Preparation

- Download and preprocess the Foursquare POI data for Pune, focusing on Hadapsar.
- Retrieve and preprocess VIIRS nighttime lights data for the region (aligned spatially and temporally).

### 2. Data Integration

- Align spatial resolutions using GIS tools or Python libraries (e.g., `geopandas`, `rasterio`).
- Merge POI and nighttime lights data using spatial joins.

### 3. Analysis

- Perform exploratory analysis to visualize POI density, lighting intensity, and correlations.
- Identify clusters, trends, and outliers (e.g., areas with high lighting but no POIs).

### 4. Visualization

- Create maps and heatmaps to illustrate findings.
- Use dynamic visualization tools (e.g., `folium`, `plotly`) to enhance interactivity.

### 5. Insights and Reporting

- Document key observations with charts, graphs, and case studies.
- Summarize insights in a report or presentation format.

---

## Tools and Technologies

- **Programming**: Python (with `pandas`, `geopandas`, `matplotlib`, `folium`, `rasterio`).
- **GIS Software**: QGIS or ArcGIS for spatial analysis.
- **Data Storage**: Amazon S3 for large datasets.
- **Visualization**: Plotly, Folium, or Kepler.gl for mapping.

---

## Expected Outcomes

- A clear understanding of the relationship between POI distribution and nighttime lights in Hadapsar.
- Visualizations and models illustrating urbanization patterns and economic activity.
- Recommendations for urban planning and development in the region.

---

## How to Use This Repository

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/hadapsar-poi-viirs-analysis.git
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Data Access**:

   - Download Foursquare POI data from [link](https://foursquare.com/dataset-link).
   - Retrieve VIIRS nighttime lights data from [XKDR](https://xkdr.org).

4. **Run Analysis**:

   - Use the provided Jupyter notebooks to preprocess, analyze, and visualize the data.
   - Explore example notebooks for step-by-step guidance.

5. **Contribute**:
   - Fork the repository, make improvements, and submit a pull request.

---

## Next Steps

- Validate initial findings through case studies and stakeholder feedback.
- Expand analysis to other Pune neighborhoods or cities.
- Publish results in open forums or research papers.

---

## Acknowledgments

- Data provided by [Foursquare](https://foursquare.com) and [VIIRS](https://xkdr.org).
- Special thanks to contributors and collaborators.
