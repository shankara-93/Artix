import os
import re
from datetime import datetime
from datetime import datetime
from typing import List, Dict, Tuple 
from PIL import Image
from PIL.ExifTags import TAGS 
import rasterio

import geopandas as gpd
from shapely.geometry import Polygon,Point 



def pixel_to_geo(x: float, y: float, image_width: int, image_height: int, geo_bbox: list) -> tuple:
    """
    Converts pixel coordinates (x, y) to/ geographical coordinates (longitude, latitude).
    """
    min_lon, min_lat, max_lon, max_lat = geo_bbox
    lon = min_lon + (x / image_width) * (max_lon - min_lon)
    lat = max_lat - (y / image_height) * (max_lat - min_lat)  # Latitude decreases as y increases
    return lon, lat

def create_geojson_feature(center_geo: tuple) -> dict:
    """
    Creates a GeoJSON feature for a given geographical coordinate (longitude, latitude).
    """
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": list(center_geo),
        },
        "properties": {
            "description": "Bounding box point",
        },
    }

def bounding_boxes_to_geodataframe(bounding_boxes: list, image_width: int, image_height: int, geo_bbox: list) -> gpd.GeoDataFrame:
    """
    Converts a list of bounding boxes into a GeoPandas GeoDataFrame, including class and confidence properties.

    Args:
        bounding_boxes (list): List of dictionaries with 'xyxy', 'class', and 'confidence' keys.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        geo_bbox (list): List containing the geographical bounding box [min_lon, min_lat, max_lon, max_lat].

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing bounding box points and their properties.
    """
    # List comprehension to create points and capture 'class' and 'confidence'
    data = [
        {
            'geometry': Point(pixel_to_geo((bbox['xyxy'][0] + bbox['xyxy'][2]) / 2,  # x_center
                                           (bbox['xyxy'][1] + bbox['xyxy'][3]) / 2,  # y_center
                                           image_width, image_height, geo_bbox)),
            'class': bbox['class'],
            'confidence': bbox['confidence']
        }
        for bbox in bounding_boxes
    ]

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(data)
    gdf.crs = "EPSG:4326"  # Set the CRS to WGS 84 (longitude/latitude)

    return gdf

########### Get image metadata #### 
def load_raster_metadata(file_path):
    with rasterio.open(file_path) as src:
        top_left_x, top_left_y = src.transform * (0, 0)  # Get top-left corner coordinates
        width_px = src.width  # Width in pixels
        height_px = src.height  # Height in pixels
        pixel_size = src.res[0]  # Pixel size in the x direction (assuming square pixels)
        crs = src.crs  # Coordinate Reference System

    return {
        'top_left_x': top_left_x,
        'top_left_y': top_left_y,
        'width_px': width_px,
        'height_px': height_px,
        'pixel_size': pixel_size,
        'crs': crs.to_string() if crs else None  # Convert CRS to string for easier reading
    }

def get_max_date_from_dict(date_file_dict: Dict[str, str]) -> str:
    """
    Returns the maximum date from the dictionary keys (dates in 'YYYY-MM-DD' format).
    
    Args:
        date_file_dict (Dict[str, str]): A dictionary with dates as keys in 'YYYY-MM-DD' format.
        
    Returns:
        str: The maximum date as a string in 'YYYY-MM-DD' format.
    """
    if not date_file_dict:
        raise ValueError("The dictionary is empty.")
    
    # Convert the keys (date strings) to datetime objects for comparison
    date_keys = [datetime.strptime(date_str, "%Y-%m-%d_%H%M%S") for date_str in date_file_dict.keys()]
    
    # Get the maximum date and convert it back to string format
    max_date = max(date_keys)
    
    return max_date.strftime("%Y-%m-%d_%H%M%S") 

######


def get_previous_dates_and_closest(input_dict, input_key):
    # Convert the date portion of the key into a datetime object for comparison
    input_date = datetime.strptime(input_key.split('_')[0], '%Y-%m-%d')

    # List to store all previous dates, including the input_key
    previous_dates = [input_key]

    # Variable to store the closest previous date (initialize with input_key)
    closest_previous_date = None
    closest_previous_key = input_key

    # Iterate over the dictionary keys and compare dates
    for key in input_dict.keys():
        date_str = key.split('_')[0]
        current_date = datetime.strptime(date_str, '%Y-%m-%d')

        # If the current date is earlier than the input date
        if current_date < input_date:
            previous_dates.append(key)

            # Update closest_previous_date if it's the most recent one before input_date
            if closest_previous_date is None or current_date > closest_previous_date:
                closest_previous_date = current_date
                closest_previous_key = key

    # If no previous dates exist (only the input_key is in previous_dates), set closest to input_key
    if len(previous_dates) == 1:
        closest_previous_key = input_key

    return previous_dates, closest_previous_key