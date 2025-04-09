# backend/pixeltogeojson.py
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def pixeltogeo(x: float, y: float, imagewidth: int, imageheight: int, geo_bbox: list) -> Tuple:
    """Converts pixel coordinates to geographical coordinates."""
    maxlat, minlon, minlat, maxlon = geo_bbox
    lon = minlon + (x / imagewidth) * (maxlon - minlon)
    lat = maxlat - (y / imageheight) * (maxlat - minlat)
    return lon, lat

def creategeojsonfeature(center_geo: Tuple) -> dict:
    """Creates a GeoJSON feature for a given geographical coordinate."""
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
    """Converts bounding boxes to a GeoDataFrame."""
    try:
        data = [{'geometry': Point(pixeltogeo((bbox['bbox'][0] + bbox['bbox'][2]) / 2,(bbox['bbox'][1] + bbox['bbox'][3]) / 2,imagewidth, imageheight, geo_bbox)),'class': bbox['class'],'confidence': bbox['confidence']} for bbox in boundingboxes] # corrected for class_name
        gdf = gpd.GeoDataFrame(data)
        gdf.crs = "EPSG:4326"
        return gdf
    except Exception as e:
        logger.exception("Error converting bounding boxes to GeoDataFrame")
        raise RuntimeError(f"Error converting to GeoDataFrame: {e}")