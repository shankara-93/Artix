# backend/parking_analysis.py
# --- Make sure all these imports are present ---
import json
from typing import Optional, Tuple, Any, Dict, List # Added List here
import osmnx as ox # <--- Make sure osmnx is imported
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from shapely.geometry import Point,MultiPoint, Polygon, MultiPolygon # Added Polygon based on usage
# Import exceptions for specific handling if needed
from requests.exceptions import RequestException
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.geocoders import Nominatim
from shapely.ops import unary_union

# --- Initialize Logger (Good practice) ---
logger = logging.getLogger(__name__)
# --- End Logger ---

# --- Configure OSMnx Settings ---
# Configure cache folder (optional, defaults to ./cache)
# ox.settings.cache_folder = './osmnx_cache'
ox.settings.use_cache = True # Enable caching
ox.settings.log_console = False # Set to True for more detailed osmnx logging if needed
logger.info(f"OSMnx caching enabled. Cache folder: {ox.settings.cache_folder}")
# --------------------------------

def categorize_parking(value):
    # Ensure value is treated as string and handle None/NaN
    value_str = str(value).lower() if pd.notna(value) else 'none'
    # Expanded categories based on common OSM tags
    if value_str in ['yes', 'public', 'street_side', 'surface', 'multi-storey', 'underground', 'none']:
        return "Public"
    elif value_str == "private":
        return "Private"
    elif value_str in ["customers", "clients"]:
        return "Commercial"
    elif value_str == "disabled":
        return "Disabled" # Add specific category
    elif value_str == "no": # Explicitly marked as not parking
        return "No Parking"
    else:
        return "Public" # Default assumption


def adjust_capacity(row):
    # Ensure area and length are numeric, handle potential NaN/None
    # Use .get() for safer access
    area = pd.to_numeric(row.get('area_m2'), errors='coerce')
    minor_length = pd.to_numeric(row.get('minor_length_m'), errors='coerce') # Example using a property

    # --- FIX: Handle NaN/Zero area ---
    if pd.isna(area) or area <= 1e-6: # Use a small tolerance instead of exact 0
        return 0 # Cannot calculate capacity without area

    # Base capacity estimate (adjust 12.50 based on local standards if needed)
    base_capacity = np.floor(area / 12.50).astype(int)

    # --- FIX: Return base_capacity if minor_length is unavailable or invalid ---
    if pd.isna(minor_length) or minor_length <= 0:
        return max(base_capacity, 0) # Ensure capacity is not negative

    # Example adjustment logic (can be refined based on parking layout analysis)
    elif minor_length <= 12: # Wide enough for standard layouts relative to area
        capacity = base_capacity
    elif 12 < minor_length < 25: # Potentially less efficient layout
        capacity = int(base_capacity * 0.80)
    else: # Very elongated, likely lower efficiency
        capacity = int(base_capacity * 0.60)

    return max(capacity, 0) # Ensure capacity is not negative after adjustments


def get_city_name_from_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> str:
    """
    Get city name using the center of a bounding box (min_lat, min_lon, max_lat, max_lon).
    Includes error handling for geocoding requests.
    """
    default_city = "Unknown Location"
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
         logger.warning(f"Invalid bbox format for city lookup: {bbox}")
         return default_city

    # Validate coordinate ranges
    min_lat, min_lon, max_lat, max_lon = bbox
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90 and \
            -180 <= min_lon <= 180 and -180 <= max_lon <= 180 and \
            min_lat <= max_lat and min_lon <= max_lon):
         logger.warning(f"Invalid coordinate values or order in bbox: {bbox}")
         return default_city


    try:
        # Increase user agent detail
        geolocator = Nominatim(user_agent="ArtixUrbanX/1.0 (ParkingAnalysis; mailto:contact@example.com)") # Replace with actual contact if applicable

        # Calculate center of bounding box
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2

        logger.debug(f"Performing reverse geocoding for: ({mid_lat}, {mid_lon})")
        # Perform reverse geocoding with timeout and language preference
        location = geolocator.reverse((mid_lat, mid_lon), exactly_one=True, timeout=10, language='en')

        if location and location.raw and 'address' in location.raw:
            address = location.raw['address']
            # Try multiple keys for city/town/etc.
            city = address.get('city') or address.get('town') or address.get('village') or address.get('municipality') or address.get('county')
            logger.debug(f"Geocoding result: {address}")
            return city if city else default_city
        else:
            logger.warning(f"Reverse geocoding returned no result or no address for coords: ({mid_lat}, {mid_lon})")
            return default_city
    except (GeocoderTimedOut, GeocoderServiceError, RequestException) as geo_err:
         # Handle specific geocoding errors
         logger.error(f"Error during reverse geocoding ({geo_err.__class__.__name__}): {geo_err}")
         return default_city # Fallback on geocoding errors
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"Unexpected error during reverse geocoding: {e}")
        return default_city # Default fallback on other errors


def add_geometric_properties(data: gpd.GeoDataFrame, properties: Optional[List[str]] = None, area_unit: str = "m2", length_unit: str = "m") -> gpd.GeoDataFrame:
    """
    Calculates geometric properties and adds them to the GeoDataFrame.
    Improved error handling and CRS management.
    """
    # --- Input Validation ---
    if not isinstance(data, gpd.GeoDataFrame):
        raise TypeError("Input must be a GeoDataFrame")
    if data.empty:
         logger.warning("Input GeoDataFrame to add_geometric_properties is empty.")
         return data # Return empty GDF if input is empty
    if data.crs is None:
        logger.warning("Input GeoDataFrame has no CRS defined. Geometric properties might be inaccurate or fail.")
        # Optionally raise ValueError or attempt to set a default CRS if appropriate
        # raise ValueError("GeoDataFrame must have a defined coordinate reference system (CRS)")
        # For now, proceed with caution
    if 'geometry' not in data.columns:
        raise ValueError("Input GeoDataFrame must have a 'geometry' column.")


    result = data.copy() # Work on a copy

    # Default properties if none specified (calculate only commonly needed ones?)
    if properties is None:
        properties = [
            "area", "length", "minor_length" # Keep minimum needed for capacity?
            # Add others back if required downstream
            # "perimeter", "convex_hull_area", "orientation",
            # "complexity", "area_bbox", "area_convex", "area_filled",
            # "major_length", "eccentricity", "diameter_areagth",
            # "extent", "solidity", "elongation"
        ]
    logger.debug(f"Calculating geometric properties: {properties}")

    original_crs = result.crs
    result_proj = result # Start with original GDF

    # --- Reproject to suitable projected CRS for accurate measurements ---
    projected_crs = None
    if original_crs and original_crs.is_geographic:
        try:
            # Estimate UTM CRS based on the GDF's extent
            projected_crs = result.estimate_utm_crs()
            if projected_crs:
                logger.info(f"Reprojecting from geographic CRS {original_crs} to estimated UTM {projected_crs} for measurements.")
                result_proj = result.to_crs(projected_crs)
            else:
                logger.warning(f"Could not estimate UTM CRS from bounds. Using original CRS {original_crs}, measurements may be inaccurate.")
                # Keep result_proj = result (original geographic CRS)
        except Exception as e:
            logger.warning(f"Failed to estimate or reproject to UTM: {e}. Using original CRS {original_crs}, measurements may be inaccurate.")
            # Keep result_proj = result (original geographic CRS)
    elif original_crs:
         # Already projected, proceed with original CRS for calculations
         logger.debug(f"Using existing projected CRS {original_crs} for measurements.")
         projected_crs = original_crs # Keep track
    else:
         # No CRS, proceed with caution
         logger.warning("No CRS defined, measurements will be in arbitrary units and potentially inaccurate.")


    # --- Define Helper for Safe Geometry Operations ---
    def safe_geom_op(geom, operation, default_value=0):
         if geom is None or geom.is_empty or not geom.is_valid:
              # Log invalid geometry?
              # logger.debug(f"Skipping operation on invalid/empty geometry: {geom}")
              return default_value
         try:
              return operation(geom)
         except Exception as op_err:
              # logger.warning(f"Error during geometry operation {operation.__name__}: {op_err} on geom {geom.wkt[:100]}...")
              return default_value

    # --- Calculate Properties on result_proj (using safe_geom_op) ---
    final_prop_names = {} # To store final column names after unit conversion

    # AREA
    if "area" in properties:
        result_proj["geom_area"] = result_proj.geometry.apply(safe_geom_op, operation=lambda g: g.area, default_value=np.nan)
        col_name = f"area_{area_unit}"
        final_prop_names["area"] = col_name
        if area_unit == "km2": result_proj[col_name] = result_proj["geom_area"] / 1_000_000
        elif area_unit == "ha": result_proj[col_name] = result_proj["geom_area"] / 10_000
        else: result_proj[col_name] = result_proj["geom_area"] # Default m2
        result_proj.drop(columns=["geom_area"], inplace=True) # Drop intermediate

    # LENGTH
    if "length" in properties:
        result_proj["geom_length"] = result_proj.geometry.apply(safe_geom_op, operation=lambda g: g.length, default_value=np.nan)
        col_name = f"length_{length_unit}"
        final_prop_names["length"] = col_name
        if length_unit == "km": result_proj[col_name] = result_proj["geom_length"] / 1_000
        else: result_proj[col_name] = result_proj["geom_length"] # Default m
        result_proj.drop(columns=["geom_length"], inplace=True)

    # PERIMETER (boundary length)
    if "perimeter" in properties:
        result_proj["geom_perimeter"] = result_proj.geometry.apply(safe_geom_op, operation=lambda g: g.boundary.length if isinstance(g, (Polygon, MultiPolygon)) else np.nan, default_value=np.nan)
        col_name = f"perimeter_{length_unit}"
        final_prop_names["perimeter"] = col_name
        if length_unit == "km": result_proj[col_name] = result_proj["geom_perimeter"] / 1_000
        else: result_proj[col_name] = result_proj["geom_perimeter"] # Default m
        result_proj.drop(columns=["geom_perimeter"], inplace=True)

    # MINIMUM ROTATED RECTANGLE based properties (Major/Minor axes etc.)
    mrr_props_needed = any(p in properties for p in ["major_length", "minor_length", "eccentricity", "orientation", "elongation"])
    if mrr_props_needed:
        def get_mrr_axes(geom):
            mrr = safe_geom_op(geom, operation=lambda g: g.minimum_rotated_rectangle, default_value=None)
            if mrr is None or not isinstance(mrr, Polygon) or mrr.is_empty:
                return pd.Series([np.nan, np.nan], index=['major_length', 'minor_length']) # Return NaNs

            try:
                coords = list(mrr.exterior.coords)[:-1]
                if len(coords) < 4: return pd.Series([np.nan, np.nan], index=['major_length', 'minor_length'])

                dists = [Point(coords[i]).distance(Point(coords[i+1])) for i in range(4)]
                side1 = dists[0]
                side2 = dists[1]

                major = max(side1, side2)
                minor = min(side1, side2)
                # Add other MRR props here if needed (eccentricity, orientation etc.)
                return pd.Series([major, minor], index=['major_length', 'minor_length'])
            except Exception as mrr_err:
                 # logger.warning(f"Error calculating MRR properties: {mrr_err}")
                 return pd.Series([np.nan, np.nan], index=['major_length', 'minor_length'])

        mrr_data = result_proj.geometry.apply(get_mrr_axes)
        # Add results to dataframe
        if "major_length" in properties:
            col_name = f"major_length_{length_unit}"
            final_prop_names["major_length"] = col_name
            result_proj[col_name] = mrr_data['major_length']
            if length_unit == "km": result_proj[col_name] /= 1000
        if "minor_length" in properties:
            col_name = f"minor_length_{length_unit}"
            final_prop_names["minor_length"] = col_name
            result_proj[col_name] = mrr_data['minor_length']
            if length_unit == "km": result_proj[col_name] /= 1000
        # Calculate and add other MRR-derived properties like eccentricity, orientation, elongation here if needed

    # ... (Add calculations for other properties like complexity, solidity, extent etc. IF they are included in `properties`)
    # Make sure to use the `final_prop_names` mapping for dependencies (e.g., extent needs area and area_bbox)

    # --- Copy calculated columns back to original CRS dataframe ---
    new_cols = [col for col in result_proj.columns if col not in result.columns and col != result_proj.geometry.name]
    logger.debug(f"Copying new columns back to original GDF: {new_cols}")
    for col in new_cols:
        # Ensure index alignment before copying
        if result.index.equals(result_proj.index):
             result[col] = result_proj[col]
        else:
             logger.warning(f"Index mismatch when copying column {col}. Attempting merge.")
             # Attempt merge, might be slow or fail if indices are not compatible
             try:
                  result = result.merge(result_proj[[col]], left_index=True, right_index=True, how='left')
             except Exception as merge_err:
                  logger.error(f"Failed to merge column {col} due to index issues: {merge_err}")
                  # Handle error - maybe fill with NaN or raise?
                  result[col] = np.nan


    # --- Final Check and Return ---
    # Ensure essential columns (like area_m2 for capacity) exist before returning
    if 'area_m2' not in result.columns and 'area' in properties:
         logger.error("Column 'area_m2' was requested but not successfully created.")
         # Optionally add it with NaNs
         # result['area_m2'] = np.nan

    return result # Return the DataFrame with original CRS + new columns


def get_parking_data(bbox: Tuple[float, float, float, float], crs: str = "EPSG:4326") -> Optional[gpd.GeoDataFrame]:
    """
    Get parking polygons from OpenStreetMap (OSM) within the given bounding box.

    Parameters:
    - bbox (tuple): Bounding box coordinates in (left, bottom, right, top) format.
    - crs (str): Coordinate Reference System (CRS) to use. Default is "EPSG:4326" (WGS84).

    Returns:
    - GeoDataFrame or None: Returns a GeoDataFrame with parking polygons if available, else None.
    """
    if len(bbox) != 4 or not all(isinstance(coord, (int, float)) for coord in bbox):
        logger.error("Invalid bounding box format. Expected (left, bottom, right, top) as float values.")
        return None

    tags = {'amenity': 'parking'}

    try:
        # Convert bbox to OSM's expected format (south, west, north, east)
        osm_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        parking_gdf = ox.features.features_from_bbox(osm_bbox, tags=tags)

        if parking_gdf.empty:
            logger.info("No parking polygons found in the given area.")
            return None

        # Ensure CRS is set correctly
        if parking_gdf.crs is None:
            parking_gdf.set_crs(crs, inplace=True)
        else:
            parking_gdf = parking_gdf.to_crs(crs)

        # Add geometric properties
        parking_gdf = add_geometric_properties(parking_gdf)

        # Categorize parking based on 'access' attribute
        parking_gdf['Category'] = parking_gdf.get('access', '').apply(categorize_parking)

        # Calculate estimated capacity (assuming 12.5 m² per parking spot)
        parking_gdf['Gross_Capacity'] = np.floor(parking_gdf['area_m2'] / 12.50).astype(int)

        # Adjust capacity based on additional parameters
        parking_gdf['Capacity'] = parking_gdf.apply(adjust_capacity, axis=1)

        # Rename index columns and reset index
        parking_gdf = parking_gdf.reset_index(level=['element', 'id'])
        parking_gdf.rename(columns={'id': 'osmid'}, inplace=True)

        # Reproject back to original CRS if necessary
        parking_gdf = parking_gdf.to_crs(crs)

        return parking_gdf

    except Exception as e:
        logger.exception("Error retrieving parking data from OSM.")
        return None


# --- intersecting_geojson (Consider minor improvements like early exits and type hints) ---
def intersecting_geojson(
    gdf_centroids: gpd.GeoDataFrame,
    gdf_parking: gpd.GeoDataFrame,
    bounds: Optional[Tuple[float, float, float, float]], # (min_lat, min_lon, max_lat, max_lon)
    crs: str = "EPSG:4326"
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any], gpd.GeoDataFrame]:
    """
    Find intersections between centroids and parking polygons, aggregate data,
    and identify non-intersecting polygons. Includes error handling and metadata calculation.
    Uses the user's original aggregation logic (MultiPoint for occupied centroids).
    """
    logger.info("Starting intersection analysis (using MultiPoint aggregation)...")
    # --- Initialize error return structure ---
    occupied_cols = ['osmid', 'area_m2', 'Category', 'Capacity', 'polygon_geometry', 'geometry', 'occupancy', 'occupancy(%)']
    occupied_gdf_empty = gpd.GeoDataFrame(columns=occupied_cols, crs=crs, geometry='geometry')
    empty_gdf_cols = ['osmid', 'geometry', 'Category', 'Capacity', 'area_m2'] # Expected columns for empty GDF
    empty_gdf_empty = gpd.GeoDataFrame(columns=empty_gdf_cols, crs=crs, geometry='geometry')

    # Default metadata structure
    default_metadata = {
        "city_name": "Unknown Area", "num_parking": 0, "parking_capacity": 0,
        "parking_area_capacity(sq.m)": 0.0, "total_parking_area(sq.m)": 0.0,
        "occupancy_per(counts)": 0.0, "occupancy_per(area)": 0.0,
        "Total Occupied (Cars)": 0, "parked_area(sq.m)": 0.0,
        "Avg Occupancy (Cars)": 0.0, "Empty Parking areas": 0,
        "Occupied Parking areas": 0, "error": None # Start with no error
    }
    # Try to get city name early, but don't let it block analysis if it fails
    default_metadata["city_name"] = get_city_name_from_bbox(bounds) if bounds else "Unknown Area"

    try:
        # --- Input Validation ---
        if not isinstance(gdf_centroids, gpd.GeoDataFrame) or not isinstance(gdf_parking, gpd.GeoDataFrame):
            msg = "Invalid input: Must be GeoDataFrames."
            logger.error(msg)
            default_metadata["error"] = msg
            return occupied_gdf_empty, default_metadata, empty_gdf_empty
        if 'geometry' not in gdf_centroids.columns or 'geometry' not in gdf_parking.columns:
            msg = "Missing 'geometry' column."
            logger.error(msg)
            default_metadata["error"] = msg
            return occupied_gdf_empty, default_metadata, empty_gdf_empty
        if 'osmid' not in gdf_parking.columns:
            msg = "Missing 'osmid' column in parking data."
            logger.error(msg)
            default_metadata["error"] = msg
            return occupied_gdf_empty, default_metadata, empty_gdf_empty

        # --- Handle Empty Inputs Gracefully ---
        total_parking_polygons = len(gdf_parking['osmid'].unique()) if not gdf_parking.empty else 0
        total_capacity = int(gdf_parking['Capacity'].sum()) if not gdf_parking.empty and 'Capacity' in gdf_parking.columns else 0
        total_area = round(gdf_parking['area_m2'].sum(), 2) if not gdf_parking.empty and 'area_m2' in gdf_parking.columns else 0.0

        default_metadata.update({
             "num_parking": total_parking_polygons, "parking_capacity": total_capacity,
             "total_parking_area(sq.m)": total_area, "Empty Parking areas": total_parking_polygons,
             "parking_area_capacity(sq.m)": round(total_capacity * 12.50, 2) # Estimate based on capacity
        })

        if gdf_centroids.empty or gdf_parking.empty:
            logger.warning("Input centroids or parking GDF is empty. No intersections possible.")
            # Return empty occupied, full parking as empty, and calculated metadata
            non_intersecting_parking = gdf_parking.copy().to_crs(crs)
            # Select relevant columns for empty gdf
            non_intersecting_parking = non_intersecting_parking[[col for col in empty_gdf_cols if col in non_intersecting_parking.columns]]
            return occupied_gdf_empty, default_metadata, non_intersecting_parking


        # --- Ensure Consistent CRS ---
        if gdf_centroids.crs != gdf_parking.crs:
            logger.warning(f"Aligning CRS: centroids {gdf_centroids.crs} -> parking {gdf_parking.crs}")
            try:
                 gdf_centroids = gdf_centroids.to_crs(gdf_parking.crs)
            except Exception as crs_err:
                 msg = f"Failed to align CRS: {crs_err}"
                 logger.error(msg, exc_info=True)
                 default_metadata["error"] = msg
                 return occupied_gdf_empty, default_metadata, empty_gdf_empty # Return empty on CRS error

        # --- Spatial Join (Points within Polygons) ---
        logger.info("Performing spatial join (predicate='within')...")
        # Note: sjoin preserves the index of the left GDF (centroids) and adds 'index_right' for the parking GDF index
        # Use .copy() to avoid modifying original GDFs if they are used elsewhere
        intersecting_points = gpd.sjoin(gdf_centroids.copy(), gdf_parking.copy(), how='inner', predicate='within')
        logger.info(f"Found {len(intersecting_points)} intersecting points (centroids within parking polygons).")

        if intersecting_points.empty:
             logger.warning("No intersections found after spatial join.")
             # Return empty occupied, full parking as empty, and calculated metadata
             non_intersecting_parking = gdf_parking.copy().to_crs(crs)
             non_intersecting_parking = non_intersecting_parking[[col for col in empty_gdf_cols if col in non_intersecting_parking.columns]]
             return occupied_gdf_empty, default_metadata, non_intersecting_parking

        # --- Add Polygon Geometry to Intersecting Points ---
        # 'index_right' column holds the original index from gdf_parking
        if 'index_right' not in intersecting_points.columns:
             msg = "Spatial join did not produce 'index_right' column."
             logger.error(msg)
             default_metadata["error"] = msg
             return occupied_gdf_empty, default_metadata, empty_gdf_empty

        try:
             # Use .loc to safely map the parking polygon geometry using the index_right
             intersecting_points['polygon_geometry'] = intersecting_points['index_right'].map(gdf_parking['geometry'])
        except KeyError as e:
             msg = f"Error mapping polygon geometry using 'index_right'. Index mismatch? {e}"
             logger.error(msg, exc_info=True)
             default_metadata["error"] = msg
             return occupied_gdf_empty, default_metadata, empty_gdf_empty

        # Drop the index_right column if no longer needed
        # intersecting_points = intersecting_points.drop(columns=['index_right'])

        # --- Aggregation by Parking Polygon ---
        logger.info("Aggregating intersecting points by parking osmid...")
        # Group by 'osmid' which should be present in intersecting_points after the join from gdf_parking
        if 'osmid' not in intersecting_points.columns:
             msg = "Column 'osmid' not found in joined data after sjoin."
             logger.error(msg)
             default_metadata["error"] = msg
             return occupied_gdf_empty, default_metadata, empty_gdf_empty

        # Ensure required columns for aggregation exist in intersecting_points
        required_agg_cols = ['area_m2', 'Category', 'Capacity', 'polygon_geometry', 'geometry'] # geometry is the centroid point
        if not all(col in intersecting_points.columns for col in required_agg_cols):
             missing = [col for col in required_agg_cols if col not in intersecting_points.columns]
             msg = f"Missing columns required for aggregation in joined data: {missing}"
             logger.error(msg)
             default_metadata["error"] = msg
             return occupied_gdf_empty, default_metadata, empty_gdf_empty

        aggregations = {
            'area_m2': 'first', # Assuming area is constant per osmid
            'Category': 'first', # Assuming category is constant per osmid
            'Capacity': 'first', # Assuming capacity is constant per osmid
            "polygon_geometry": 'first', # Get the parking polygon geometry
            'geometry': lambda pts: MultiPoint([p for p in pts if p is not None and not p.is_empty]) # Aggregate centroid points
        }

        # Perform the aggregation, grouping by the parking feature's osmid
        grouped_df = intersecting_points.groupby('osmid', as_index=False).agg(aggregations)
        # Calculate occupancy count per group
        occupancy_counts = intersecting_points.groupby('osmid').size()
        grouped_df = grouped_df.merge(occupancy_counts.rename('occupancy'), on='osmid')


        # Convert back to GeoDataFrame, geometry is now MultiPoint (centroids)
        grouped_gdf = gpd.GeoDataFrame(grouped_df, geometry='geometry', crs=gdf_parking.crs) # Use original CRS before final conversion
        logger.info(f"Aggregation complete. Created {len(grouped_gdf)} occupied parking entries.")

        # --- Calculate Occupancy Percentage ---
        # Ensure types are correct before calculation
        grouped_gdf['area_m2'] = pd.to_numeric(grouped_gdf['area_m2'], errors='coerce').fillna(0).round(2)
        grouped_gdf['Capacity'] = pd.to_numeric(grouped_gdf['Capacity'], errors='coerce').fillna(0).astype(int)
        grouped_gdf['occupancy'] = pd.to_numeric(grouped_gdf['occupancy'], errors='coerce').fillna(0).astype(int)

        # Calculate occupancy percentage, handling zero capacity
        grouped_gdf['occupancy(%)'] = (
            (grouped_gdf['occupancy'] / grouped_gdf['Capacity'].replace(0, np.nan)) * 100
        ).fillna(0).round(2) # Fill NaN resulting from Capacity=0 with 0%

        # --- Reproject Occupied GDF to Target CRS ---
        logger.info(f"Converting occupied GDF to target CRS: {crs}")
        occupied_gdf_final = grouped_gdf.to_crs(crs)

        # --- Identify Non-Intersecting Parking Polygons ---
        logger.info("Identifying non-intersecting parking polygons...")
        # Get indices of parking polygons that *did* intersect (using 'index_right' from the join)
        intersecting_parking_indices = intersecting_points['index_right'].unique()
        # Select polygons from the original gdf_parking whose index is NOT in the intersecting set
        non_intersecting_parking = gdf_parking[~gdf_parking.index.isin(intersecting_parking_indices)].copy()
        non_intersecting_parking = non_intersecting_parking.to_crs(crs) # Convert to target CRS
        # Select relevant columns for empty gdf
        non_intersecting_parking = non_intersecting_parking[[col for col in empty_gdf_cols if col in non_intersecting_parking.columns]]

        logger.info(f"Found {len(non_intersecting_parking)} non-intersecting polygons.")

        # --- Final Metadata Calculation ---
        logger.info("Calculating final metadata...")
        total_occupied_cars = occupied_gdf_final['occupancy'].sum()
        num_occupied_polygons = len(occupied_gdf_final)
        num_empty_polygons = len(non_intersecting_parking)
        estimated_occupied_area = total_occupied_cars * 12.50 # Estimate based on count

        # Use pre-calculated totals for percentages
        occupancy_count_percent = (total_occupied_cars / total_capacity * 100) if total_capacity > 0 else 0
        # Use actual total area if available for area percentage
        occupancy_area_percent = (estimated_occupied_area / total_area * 100) if total_area > 0 else 0

        final_metadata = default_metadata.copy() # Start with defaults
        final_metadata.update({
            "occupancy_per(counts)": round(occupancy_count_percent, 2),
            "occupancy_per(area)": round(occupancy_area_percent, 2),
            "Total Occupied (Cars)": int(total_occupied_cars),
            "parked_area(sq.m)": round(estimated_occupied_area, 2),
            "Avg Occupancy (Cars)": round(occupied_gdf_final['occupancy'].mean(), 2) if num_occupied_polygons > 0 else 0.0,
            "Empty Parking areas": num_empty_polygons,
            "Occupied Parking areas": num_occupied_polygons,
            "error": None # Explicitly set error to None on success
        })
        logger.info("Metadata calculation complete.")

        # --- Select and Reorder Columns for Consistency ---
        # Ensure required columns exist before selecting
        occupied_gdf_final = occupied_gdf_final[[col for col in occupied_cols if col in occupied_gdf_final.columns]]
        # non_intersecting_parking columns selected earlier

        logger.info("Intersection analysis function finished successfully.")
        return occupied_gdf_final, final_metadata, non_intersecting_parking

    except Exception as e:
        logger.exception(f"!!! UNEXPECTED ERROR in intersecting_geojson: {e}")
        error_metadata = default_metadata.copy()
        error_metadata["error"] = f"Internal error during analysis: {e.__class__.__name__}: {str(e)}"
        # Return initialized empty GDFs and error metadata
        return occupied_gdf_empty, error_metadata, empty_gdf_empty


# --- multipoint_to_dataframe (Keep as is, no major performance concerns likely) ---
def multipoint_to_dataframe(geo_dataframe):
    """
    Converts a GeoDataFrame containing MultiPoint or Point geometries into a pandas DataFrame
    with columns for longitude, latitude, and a weight column indicating the number of points
    in each MultiPoint geometry.
    """
    locations_parking = []
    if 'geometry' not in geo_dataframe.columns:
         logger.warning("multipoint_to_dataframe called with GDF missing 'geometry' column.")
         return pd.DataFrame(columns=["longitude", "latitude", "weight"])

    for geom in geo_dataframe.geometry:
        # Add checks for None or empty geometry
        if geom is None or geom.is_empty: continue

        if geom.geom_type == 'MultiPoint':
            num_points = len(geom.geoms)
            for point in geom.geoms:
                 # Check individual points
                 if point is not None and not point.is_empty:
                      locations_parking.append([point.x, point.y, num_points])
        elif geom.geom_type == 'Point':
            # Assign weight 1 for single points
            locations_parking.append([geom.x, geom.y, 1])
        # else: ignore other geometry types silently? Or log warning?
        #    logger.debug(f"Ignoring non-Point/MultiPoint geometry type {geom.geom_type} in multipoint_to_dataframe")


    df_locations_parking = pd.DataFrame(locations_parking, columns=["longitude", "latitude", "weight"])
    return df_locations_parking