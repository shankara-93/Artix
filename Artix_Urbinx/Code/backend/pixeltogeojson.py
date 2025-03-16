import geopandas as gpd 
from typing import List, Dict,Tuple
from shapely.geometry import Point



def pixeltogeo(x: float, y: float, imagewidth: int, imageheight: int, geo_bbox: list) -> Tuple:
    """
    Converts pixel coordinates (x, y) to geographical coordinates (longitude, latitude),
    while correctly interpreting the geo_bbox in the format [maxlat, minlon, minlat, maxlon].
    """
    maxlat, minlon, minlat, maxlon = geo_bbox  # Corrected to match input format
    lon = minlon + (x / imagewidth) * (maxlon - minlon)
    lat = maxlat - (y / imageheight) * (maxlat - minlat)  # Latitude decreases as y increases
    return lon, lat

def creategeojsonfeature(center_geo: Tuple) -> dict:
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

def boundingboxestogeodataframe(boundingboxes: list, imagewidth: int, imageheight: int, geo_bbox: list) -> gpd.GeoDataFrame:
    """
    Converts a list of bounding boxes into a GeoPandas GeoDataFrame, including class and confidence properties.

    Args:
        bounding_boxes (list): List of dictionaries with 'xyxy', 'class', and 'confidence' keys.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
        geo_bbox (list): List containing the geographical bounding box [maxlat, minlon, minlat, maxlon].

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing bounding box points and their properties.
    """
    # List comprehension to create points and capture 'class' and 'confidence'
    data = [{'geometry': Point(pixeltogeo((bbox['bbox'][0] + bbox['bbox'][2]) / 2,(bbox['bbox'][1] + bbox['bbox'][3]) / 2,imagewidth, imageheight, geo_bbox)),'class': bbox['class'],'confidence': bbox['confidence']} for bbox in boundingboxes]

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(data)
    gdf.crs = "EPSG:4326"  # Set the CRS to WGS 84 (longitude/latitude)

    return gdf
