import json
import osmnx as ox
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_bbox_to_nsew(minx, miny, maxx, maxy):
    """
    Convert bounding box coordinates to north, south, east, west.

    Parameters:
    - minx (float): Minimum longitude.
    - miny (float): Minimum latitude.
    - maxx (float): Maximum longitude.
    - maxy (float): Maximum latitude.

    Returns:
    - tuple: (north, south, east, west)
    """
    return maxy, miny, maxx, minx

def categorize_parking(value):
    if value in ["yes", "public"]:
        return "Public"
    elif value == "private":
        return "Private"
    elif value in ["customers", "commercial"]:
        return "Commercial"
    else:
        return "Public"  # Default case

def convert_to_bounds(coord_list):
    """
    Converts a list of coordinates [lat_max, lon_min, lat_min, lon_max]
    into bounds (lon_max, lat_min, lon_min, lat_max).

    Args:
    - coord_list (list): A list of coordinates [lat_max, lon_min, lat_min, lon_max]

    Returns:
    - tuple: A tuple representing bounds (lon_max, lat_min, lon_min, lat_max)
    """
    lat_max, lon_min, lat_min, lon_max = coord_list
    bounds = (lon_max, lat_min, lon_min, lat_max)
    return bounds

def download_parking_data(bounds, crs="EPSG:4326"):
    """
    Download parking polygons from OSM within the bounding box and return as GeoJSON.

    Parameters:
    - bounds (tuple): Bounding box coordinates (minx, miny, maxx, maxy).
    - crs (str): Coordinate Reference System (CRS) to use. Default is "EPSG:4326" (WGS84).

    Returns:
    - str: GeoJSON data as a string.
    """
    # Convert bounding box to north, south, east, west format
    north, south, east, west = convert_bbox_to_nsew(*convert_to_bounds(bounds))
    tags = {'amenity': 'parking'}

    try:
        # Download parking polygons from OSM
        parking_gdf = ox.geometries_from_bbox(north, south, east, west, tags)

        if parking_gdf.empty:
            logging.info("No parking polygons found in the given area.")
            return None

        # Set CRS if not already set
        if parking_gdf.crs is None:
            parking_gdf.set_crs(crs, inplace=True)
        else:
            parking_gdf = parking_gdf.to_crs(crs)  

        # Calculate the area of polygons
        parking_gdf = parking_gdf.to_crs(epsg=32633)  # Project to metric CRS for area calculation
        parking_gdf['area_m2'] = parking_gdf.geometry.area
        parking_gdf['Category'] = parking_gdf['access'].apply(categorize_parking)
        parking_gdf['Capacity'] = (parking_gdf['area_m2'] /12.50).apply(np.floor).astype(int)
        parking_gdf= parking_gdf.reset_index(level=['element_type', 'osmid']) 
        parking_gdf = parking_gdf.to_crs(crs)  # Reproject back to original CRS
        
        return parking_gdf

    except Exception as e:
        logging.error(f"Error: {e}")
        logging.exception("Full traceback:")
        return None


def intersecting_geojson(gdf_centroids, gdf_parking, crs="EPSG:4326"):
    """
    Find intersections between centroids and parking polygons and aggregate data.
    """
    logging.info("Called intersecting_geojson")

    # Ensure that inputs are GeoDataFrames
    #gdf_centroids = json_geojson(gdf_centroids_dict)
    #gdf_parking = json_geojson(gdf_parking_dict)

    # Perform spatial join to find intersecting points
    intersecting_points = gpd.sjoin(gdf_centroids, gdf_parking, how='inner', predicate='intersects') 
    # Merge the polygon geometry using the correct index reference, likely 'index_right'
    intersecting_points['polygon_geometry'] = gdf_parking.loc[intersecting_points['index_right'], 'geometry'].values
    # Reset index if needed
    intersecting_points.reset_index(drop=True, inplace=True)

    # Create a new column to store the polygon geometry from df_parking
    intersecting_points['polygon_geometry'] = gdf_parking.loc[intersecting_points.index_right, 'geometry'].values

    # Aggregate by 'osmid'
    aggregations = {
        'area_m2': 'first',  # Take the first value (assuming consistent within groups)
        'Category': 'first',  # Assuming consistent category for each osmid
        'Capacity': 'first',  # Take the first capacity value for each osmid
        "polygon_geometry":'first',## Take the first polygon_geometry for each osmid
        'geometry': lambda x: MultiPoint([point for point in x])  # Combine geometries into MultiPoint
    }

    # Group by 'osmid' and aggregate the columns
    grouped_df = intersecting_points.groupby('osmid').agg(aggregations).reset_index()

    # Convert grouped_df back to GeoDataFrame
    grouped_gdf = gpd.GeoDataFrame(grouped_df, geometry='geometry', crs=gdf_centroids.crs)

    # Add the total number of points for each 'osmid'
    grouped_gdf['occupancy'] = intersecting_points.groupby('osmid')['geometry'].count().values
    grouped_gdf['area_m2'] = grouped_gdf['area_m2'].round(2) 

    # Reproject the GeoDataFrame to the desired CRS
    parking_gdf = grouped_gdf.to_crs(crs)
    
    parking_meta_data= {"num_parking":len(gdf_parking['osmid'].unique()),"parking_capacity":gdf_parking['Capacity'].sum(),"parking_area_capacity":gdf_parking['Capacity'].sum()*12.50}
    return parking_gdf,parking_meta_data 