# --- Make sure all these imports are present ---
import json
from typing import Optional, Tuple,Any
import osmnx as ox
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from shapely.geometry import Point,MultiPoint, Polygon, MultiPolygon # Added Polygon based on usage
from geopy.geocoders import Nominatim
from shapely.ops import unary_union

# --- End Imports ---

# --- Initialize Logger (Good practice) ---
logger = logging.getLogger(__name__)
# --- End Logger ---

# --- Functions exactly as provided by you ---

def categorize_parking(value):
    # Ensure value is treated as string and handle None/NaN
    value_str = str(value).lower() if pd.notna(value) else 'none'
    if value_str in ["yes", "public", 'none']: # Treat None/NaN as Public for default
        return "Public"
    elif value_str == "private":
        return "Private"
    elif value_str in ["customers", "commercial"]:
        return "Commercial"
    else:
        return "Public" # Default for other values


def adjust_capacity(row):
    # Ensure area and length are numeric, handle potential NaN/None
    area = pd.to_numeric(row.get('area_m2'), errors='coerce')
    minor_length = pd.to_numeric(row.get('minor_length_m'), errors='coerce')

    if pd.isna(area) or area <= 0:
        return 0 # Cannot calculate capacity without area

    base_capacity = np.floor(area / 12.50).astype(int)

    if pd.isna(minor_length): # If length calculation failed, use base
        return base_capacity
    elif minor_length <= 12:
        return base_capacity
    elif 12 < minor_length < 25:
        return int(base_capacity * 0.80)
    else: # minor_length >= 25
        return int(base_capacity * 0.60)


def get_city_name_from_bbox(bbox):
    """
    Get city name using the center of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates (min_lat, min_lon, max_lat, max_lon)
                      Note: This expects a different order than image_bounding_box!
                      Needs adjustment based on actual input from main.py.

    Returns:
        str: City name if found, otherwise "City Name"
    """
    # --- WARNING: Potential Bbox Order Mismatch ---
    # The docstring says (min_lat, min_lon, max_lat, max_lon)
    # The image_bounding_box from previous steps is [top_lat, min_lon, bottom_lat, max_lon]
    # Need to ensure consistency or adjust the unpacking here based on what's passed
    # Assuming for now the input *will be* (min_lat, min_lon, max_lat, max_lon) as per docstring
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
         logger.warning(f"Invalid bbox format for city lookup: {bbox}")
         return "Unknown City" # Use a clearer default

    min_lat, min_lon, max_lat, max_lon = bbox # As per docstring
    # --- End Warning ---

    try:
        geolocator = Nominatim(user_agent="parking_analysis_api_v2") # Increment user agent version

        # Calculate center of bounding box
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2

        # Perform reverse geocoding
        location = geolocator.reverse((mid_lat, mid_lon), exactly_one=True, timeout=10) # Add timeout

        if location and location.raw and 'address' in location.raw:
            # Extract city name
            address = location.raw['address']
            city = address.get('city') or address.get('town') or address.get('village') or address.get('municipality')
            return city if city else "Unknown Location" # Clearer default
        else:
            logger.warning(f"Reverse geocoding failed for coords: ({mid_lat}, {mid_lon})")
            return "Unknown Location"
    except Exception as e:
        logger.error(f"Error during reverse geocoding: {e}")
        return "Unknown Location" # Default fallback on error


def add_geometric_properties(data, properties=None, area_unit="m2", length_unit="m"):
    """Calculates geometric properties and adds them to the GeoDataFrame."""

    # Assuming read_vector is not needed as data is passed as GDF
    # if isinstance(data, str):
    #     data = read_vector(data)

    # Make a copy to avoid modifying the original
    result = data.copy()

    # Default properties to calculate if none provided
    if properties is None:
        properties = [
            "area", "length", "perimeter", "convex_hull_area", "orientation",
            "complexity", "area_bbox", "area_convex", "area_filled",
            "major_length", "minor_length", "eccentricity", "diameter_areagth", # Consistent name
            "extent", "solidity", "elongation"
        ]

    if not isinstance(result, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")
    if result.crs is None:
        raise ValueError("GeoDataFrame must have a defined coordinate reference system (CRS)")

    original_crs = result.crs # Store original CRS
    result_proj = result # Start with original

    # Ensure projected CRS for accurate measurements
    if result.crs.is_geographic:
        logger.info(f"Reprojecting from geographic CRS {original_crs} to estimated UTM for measurements.")
        try:
            projected_crs = result.estimate_utm_crs()
            if projected_crs is None:
                raise ValueError("Could not estimate UTM CRS.")
            result_proj = result.to_crs(projected_crs)
        except Exception as e:
            logger.warning(f"Failed to reproject to UTM: {e}. Calculations might be inaccurate.")
            # Proceed with original potentially geographic CRS, results may vary

    # --- Calculate Properties on result_proj ---
    # (Keeping the logic exactly as you provided, including potential renames)

    # Basic area calculation with unit conversion
    if "area" in properties:
        result_proj["area"] = result_proj.geometry.apply(
            lambda geom: geom.area if isinstance(geom, (Polygon, MultiPolygon)) else 0
        )
        if area_unit == "km2":
            result_proj["area"] = result_proj["area"] / 1_000_000
            result_proj.rename(columns={"area": "area_km2"}, inplace=True)
        elif area_unit == "ha":
            result_proj["area"] = result_proj["area"] / 10_000
            result_proj.rename(columns={"area": "area_ha"}, inplace=True)
        else:
            result_proj.rename(columns={"area": "area_m2"}, inplace=True)

    # Length calculation with unit conversion
    if "length" in properties:
        result_proj["length"] = result_proj.geometry.length
        if length_unit == "km":
            result_proj["length"] = result_proj["length"] / 1_000
            result_proj.rename(columns={"length": "length_km"}, inplace=True)
        else:
            result_proj.rename(columns={"length": "length_m"}, inplace=True)

    # Perimeter calculation
    if "perimeter" in properties:
        result_proj["perimeter"] = result_proj.geometry.apply(
            lambda geom: (geom.boundary.length if isinstance(geom, (Polygon, MultiPolygon)) else 0)
        )
        if length_unit == "km":
            result_proj["perimeter"] = result_proj["perimeter"] / 1_000
            result_proj.rename(columns={"perimeter": "perimeter_km"}, inplace=True)
        else:
            result_proj.rename(columns={"perimeter": "perimeter_m"}, inplace=True)

    # Centroid coordinates
    if "centroid_x" in properties or "centroid_y" in properties:
        centroids = result_proj.geometry.centroid
        if "centroid_x" in properties: result_proj["centroid_x"] = centroids.x
        if "centroid_y" in properties: result_proj["centroid_y"] = centroids.y

    # Bounding box properties
    if "bounds" in properties:
        bounds = result_proj.geometry.bounds
        result_proj["minx"], result_proj["miny"], result_proj["maxx"], result_proj["maxy"] = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy

    # Area of bounding box
    if "area_bbox" in properties:
        bounds = result_proj.geometry.bounds
        result_proj["area_bbox"] = (bounds.maxx - bounds.minx) * (bounds.maxy - bounds.miny)
        # Unit conversion... (same logic as area)
        area_bbox_col_name = f"area_bbox_{area_unit}"
        if area_unit == "km2": result_proj["area_bbox"] /= 1_000_000
        elif area_unit == "ha": result_proj["area_bbox"] /= 10_000
        result_proj.rename(columns={"area_bbox": area_bbox_col_name}, inplace=True)


    # Area of convex hull
    if "area_convex" in properties or "convex_hull_area" in properties:
        result_proj["area_convex"] = result_proj.geometry.convex_hull.area
        # Unit conversion...
        area_convex_col_name = f"area_convex_{area_unit}"
        if area_unit == "km2": result_proj["area_convex"] /= 1_000_000
        elif area_unit == "ha": result_proj["area_convex"] /= 10_000
        result_proj.rename(columns={"area_convex": area_convex_col_name}, inplace=True)

        # Backward compatibility handled by presence of both keys check above


    # Area of filled geometry
    if "area_filled" in properties:
        def get_filled_area(geom):
            # ... (logic as provided) ...
            if not isinstance(geom, (Polygon, MultiPolygon)): return 0
            if isinstance(geom, MultiPolygon):
                filled_polys = [Polygon(p.exterior) for p in geom.geoms if p.exterior] # check exterior exists
                return unary_union(filled_polys).area if filled_polys else 0
            else:
                return Polygon(geom.exterior).area if geom.exterior else 0

        result_proj["area_filled"] = result_proj.geometry.apply(get_filled_area)
        # Unit conversion...
        area_filled_col_name = f"area_filled_{area_unit}"
        if area_unit == "km2": result_proj["area_filled"] /= 1_000_000
        elif area_unit == "ha": result_proj["area_filled"] /= 10_000
        result_proj.rename(columns={"area_filled": area_filled_col_name}, inplace=True)


    # Axes lengths, eccentricity, orientation, elongation
    if any(p in properties for p in ["major_length", "minor_length", "eccentricity", "orientation", "elongation"]):
        def get_axes_properties(geom):
            # ... (logic as provided, includes handling MultiPolygon by largest) ...
             if not isinstance(geom, (Polygon, MultiPolygon)) or geom.is_empty: return None, None, None, None, None
             if isinstance(geom, MultiPolygon):
                  try: geom = max(geom.geoms, key=lambda p: p.area)
                  except ValueError: return None, None, None, None, None
             try:
                  rect = geom.minimum_rotated_rectangle
                  if rect.is_empty or not isinstance(rect, Polygon): return None, None, None, None, None
                  coords = list(rect.exterior.coords)[:-1]
                  if len(coords) < 4: return None, None, None, None, None
                  sides = []
                  for i in range(4):
                     p1, p2 = coords[i], coords[(i + 1) % 4]
                     dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                     length = np.sqrt(dx**2 + dy**2)
                     angle = np.degrees(np.arctan2(dy, dx)) % 180
                     sides.append((length, angle, p1, p2)) # Keep points if needed later

                  sides_grouped = {}
                  tolerance = 1e-6
                  for s in sides:
                     length, angle = s[0], s[1]
                     matched = False
                     for key in sides_grouped:
                         if abs(length - key) < tolerance:
                             sides_grouped[key].append(s); matched = True; break
                     if not matched: sides_grouped[length] = [s]

                  unique_lengths = sorted(sides_grouped.keys(), reverse=True)

                  if len(unique_lengths) == 2:
                     major_length = unique_lengths[0]; minor_length = unique_lengths[1]
                     orientation = sides_grouped[major_length][0][1]
                  elif len(unique_lengths) == 1: # Square
                     major_length = minor_length = unique_lengths[0]
                     orientation = sides_grouped[major_length][0][1]
                  else: # Fallback
                     bounds = rect.bounds
                     width = bounds[2] - bounds[0]; height = bounds[3] - bounds[1]
                     major_length = max(width, height); minor_length = min(width, height)
                     orientation = 0 if width >= height else 90

                  eccentricity = np.sqrt(1 - (minor_length**2 / major_length**2)) if major_length > 0 else 0
                  elongation = major_length / minor_length if minor_length > 0 else 1

                  return major_length, minor_length, eccentricity, orientation, elongation
             except Exception as e:
                  logger.warning(f"Error calculating axes properties: {e}")
                  return None, None, None, None, None

        axes_data = result_proj.geometry.apply(get_axes_properties)
        # Apply results back to dataframe... (same logic as provided)
        prop_map = {"major_length": 0, "minor_length": 1, "eccentricity": 2, "orientation": 3, "elongation": 4}
        for prop, idx in prop_map.items():
             if prop in properties:
                  result_proj[prop] = axes_data.apply(lambda x: x[idx] if x else None)
                  if prop in ["major_length", "minor_length"]:
                      col_name = f"{prop}_{length_unit}"
                      if length_unit == "km": result_proj[prop] /= 1000
                      result_proj.rename(columns={prop: col_name}, inplace=True)


    # Equivalent diameter based on area (using the renamed area column)
    area_col_name = f"area_{area_unit}" # Get the final area column name
    if "diameter_areagth" in properties and area_col_name in result_proj.columns:
        def get_equivalent_diameter(area):
             if pd.isna(area) or area <= 0: return None
             return 2 * np.sqrt(area / np.pi) # Area already in correct units

        # Calculate diameter based on the area column already calculated
        result_proj["equivalent_diameter_area"] = result_proj[area_col_name].apply(get_equivalent_diameter)
        # Rename based on length unit
        dia_col_name = f"equivalent_diameter_area_{length_unit}"
        if length_unit == "km": result_proj["equivalent_diameter_area"] /= 1000
        result_proj.rename(columns={"equivalent_diameter_area": dia_col_name}, inplace=True)

    # Extent
    area_bbox_col_name = f"area_bbox_{area_unit}"
    if "extent" in properties and area_col_name in result_proj.columns and area_bbox_col_name in result_proj.columns:
        result_proj["extent"] = (result_proj[area_col_name] / result_proj[area_bbox_col_name].replace(0, np.nan)).fillna(0)


    # Solidity
    area_convex_col_name = f"area_convex_{area_unit}"
    if "solidity" in properties and area_col_name in result_proj.columns and area_convex_col_name in result_proj.columns:
        result_proj["solidity"] = (result_proj[area_col_name] / result_proj[area_convex_col_name].replace(0, np.nan)).fillna(0)


    # Complexity
    perimeter_col_name = f"perimeter_{length_unit}"
    if "complexity" in properties and perimeter_col_name in result_proj.columns and area_col_name in result_proj.columns:
        # Ensure area is not zero before division and sqrt
        safe_area = result_proj[area_col_name].replace(0, np.nan)
        result_proj["complexity"] = (result_proj[perimeter_col_name] / (2 * np.sqrt(np.pi * safe_area))).fillna(0)


    # --- Copy calculated columns back to original CRS dataframe ---
    # Get columns added in projected GDF
    new_cols = [col for col in result_proj.columns if col not in result.columns and col != result_proj.geometry.name]
    # Add these columns to the original `result` DataFrame (which has original CRS)
    # Note: values are from projected calculations but assigned back to original CRS GDF
    for col in new_cols:
        result[col] = result_proj[col]

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
        osm_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
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

# def intersecting_geojson(gdf_centroids: gpd.GeoDataFrame, gdf_parking: gpd.GeoDataFrame, bounds, crs="EPSG:4326") -> tuple[gpd.GeoDataFrame, dict, gpd.GeoDataFrame]:
#     """
#     Find intersections between centroids and parking polygons, aggregate data,
#     and identify non-intersecting parking polygons. Includes robust error handling.
#     """
#     logger.info("Starting intersection analysis...")
#     logger.debug(f"Input centroids CRS: {gdf_centroids.crs}, Input parking CRS: {gdf_parking.crs}, Target CRS: {crs}")
#     logger.debug(f"Centroids empty: {gdf_centroids.empty}, Parking empty: {gdf_parking.empty}")

#     # --- Initialize error return structure ---
#     # Create empty GeoDataFrames with the target CRS early on
#     # Define expected columns for occupied gdf even when empty
#     occupied_cols = gdf_parking.columns.tolist() + ['occupancy', 'occupancy(%)'] if 'geometry' in gdf_parking.columns else ['occupancy', 'occupancy(%)']
#     occupied_gdf_empty = gpd.GeoDataFrame(columns=occupied_cols, crs=crs)
#     empty_gdf_empty = gpd.GeoDataFrame(columns=gdf_parking.columns, crs=crs) if 'geometry' in gdf_parking.columns else gpd.GeoDataFrame(crs=crs)

#     error_metadata = {
#         "city_name": get_city_name_from_bbox(bounds) if bounds else "Unknown Area",
#         "error": "An unexpected error occurred during intersection analysis.",
#         # Include other default metadata fields with 0 values if desired
#         "num_parking": 0, "parking_capacity": 0, "parking_area_capacity(sq.m)": 0,
#         "total_parking_area(sq.m)": 0, "occupancy_per(counts)": 0, "occupancy_per(area)": 0,
#         "Total Occupied (Cars)": 0, "parked_area(sq.m)": 0, "Avg Occupancy (Cars)": 0,
#         "Empty Parking areas": 0, "Occupied Parking areas": 0
#     }

#     try:
#         # --- Input Validation ---
#         if not isinstance(gdf_centroids, gpd.GeoDataFrame) or not isinstance(gdf_parking, gpd.GeoDataFrame):
#             logger.error("Inputs to intersecting_geojson must be GeoDataFrames.")
#             error_metadata["error"] = "Invalid input types (must be GeoDataFrames)."
#             return occupied_gdf_empty, error_metadata, empty_gdf_empty

#         # Ensure geometry columns exist
#         if 'geometry' not in gdf_centroids.columns or 'geometry' not in gdf_parking.columns:
#              logger.error("Input GeoDataFrames must have a 'geometry' column.")
#              error_metadata["error"] = "Missing 'geometry' column in input GeoDataFrames."
#              # Return empty GDFs matching the structure but without geometry column if input lacks it
#              return gpd.GeoDataFrame(), error_metadata, gpd.GeoDataFrame()


#         if 'osmid' not in gdf_parking.columns:
#             logger.error("gdf_parking must contain 'osmid' column.")
#             error_metadata["error"] = "Missing 'osmid' column in parking data."
#             return occupied_gdf_empty, error_metadata, empty_gdf_empty # Use initialized empty GDFs

#         # --- Data Preparation ---
#         logger.debug("Preparing parking data columns...")
#         gdf_parking = gdf_parking.copy() # Avoid SettingWithCopyWarning
#         if 'Capacity' not in gdf_parking.columns:
#             logger.warning("Parking data missing 'Capacity' column, assuming 0.")
#             gdf_parking['Capacity'] = 0
#         else:
#             gdf_parking['Capacity'] = pd.to_numeric(gdf_parking['Capacity'], errors='coerce').fillna(0)

#         if 'area_m2' not in gdf_parking.columns:
#             logger.warning("Parking data missing 'area_m2' column, calculating from geometry if possible.")
#             try:
#                 # Ensure we are in a projected CRS for meaningful area calculation
#                 gdf_parking_proj = gdf_parking.to_crs(gdf_parking.estimate_utm_crs())
#                 gdf_parking['area_m2'] = gdf_parking_proj.geometry.area
#             except Exception as area_calc_err:
#                  logger.warning(f"Could not calculate area_m2 from geometry: {area_calc_err}. Assuming 0.")
#                  gdf_parking['area_m2'] = 0
#         else:
#              gdf_parking['area_m2'] = pd.to_numeric(gdf_parking['area_m2'], errors='coerce').fillna(0)

#         if 'Category' not in gdf_parking.columns:
#             gdf_parking['Category'] = 'Unknown'
#         logger.debug("Parking data preparation complete.")

#         # --- Calculate Initial Metadata (Based on potentially modified gdf_parking) ---
#         total_parking_polygons = len(gdf_parking['osmid'].unique()) if not gdf_parking.empty else 0
#         total_capacity = gdf_parking['Capacity'].sum() if not gdf_parking.empty else 0
#         total_area = gdf_parking['area_m2'].sum() if not gdf_parking.empty else 0
#         city_name = get_city_name_from_bbox(bounds) if bounds else "Unknown Area"

#         parking_meta_data = {
#             "city_name": city_name, "num_parking": total_parking_polygons,
#             "parking_capacity": int(total_capacity),
#             "parking_area_capacity(sq.m)": round(total_capacity * 12.50, 2),
#             "total_parking_area(sq.m)": round(total_area, 2),
#             "occupancy_per(counts)": 0.0, "occupancy_per(area)": 0.0,
#             "Total Occupied (Cars)": 0, "parked_area(sq.m)": 0.0,
#             "Avg Occupancy (Cars)": 0.0, "Empty Parking areas": total_parking_polygons,
#             "Occupied Parking areas": 0, # Default to 0 occupied
#             "error": None # Placeholder for potential errors later
#         }

#         # --- Handle Empty Inputs ---
#         if gdf_centroids.empty or gdf_parking.empty:
#             logger.warning("One or both input GeoDataFrames for intersection are empty. Returning early.")
#             empty_gdf = gdf_parking.copy().to_crs(crs)
#             if not empty_gdf.empty:
#                  empty_gdf = empty_gdf.set_geometry("geometry")
#             # Use the structure of occupied_gdf_empty but ensure columns match prepared gdf_parking
#             occupied_cols_now = gdf_parking.columns.tolist() + ['occupancy', 'occupancy(%)']
#             occupied_gdf = gpd.GeoDataFrame(columns=occupied_cols_now, crs=crs)
#             return occupied_gdf, parking_meta_data, empty_gdf # Return valid tuple

#         # --- CRS Alignment ---
#         logger.debug("Aligning CRS...")
#         if gdf_centroids.crs != gdf_parking.crs:
#             logger.warning(f"Reprojecting centroids from {gdf_centroids.crs} to {gdf_parking.crs} for intersection.")
#             try:
#                 gdf_centroids = gdf_centroids.to_crs(gdf_parking.crs)
#                 logger.debug("Centroids reprojected successfully.")
#             except Exception as e:
#                 logger.error(f"CRS reprojection failed: {e}", exc_info=True)
#                 parking_meta_data["error"] = f"CRS reprojection failed: {e}"
#                 empty_gdf = gdf_parking.copy().to_crs(crs) # Return all as empty
#                 if not empty_gdf.empty:
#                      empty_gdf = empty_gdf.set_geometry("geometry")
#                 occupied_cols_now = gdf_parking.columns.tolist() + ['occupancy', 'occupancy(%)']
#                 occupied_gdf = gpd.GeoDataFrame(columns=occupied_cols_now, crs=crs)
#                 return occupied_gdf, parking_meta_data, empty_gdf # Return valid tuple

#         # --- Intersection (Spatial Join) ---
#         logger.info("Performing spatial join (centroids within parking)...")
#         # Ensure geometries are valid before join (can be slow)
#         # gdf_centroids = gdf_centroids[gdf_centroids.is_valid]
#         # gdf_parking = gdf_parking[gdf_parking.is_valid]
#         intersecting_points = gpd.sjoin(gdf_centroids, gdf_parking, how='inner', predicate='within')
#         logger.info(f"Spatial join complete. Found {len(intersecting_points)} intersections.")

#         # --- Handle No Intersections ---
#         if intersecting_points.empty:
#             logger.warning("No intersections found between centroids and parking polygons.")
#             empty_gdf = gdf_parking.copy().to_crs(crs)
#             if not empty_gdf.empty:
#                  empty_gdf = empty_gdf.set_geometry("geometry")
#             occupied_cols_now = gdf_parking.columns.tolist() + ['occupancy', 'occupancy(%)']
#             occupied_gdf = gpd.GeoDataFrame(columns=occupied_cols_now, crs=crs)
#             # Metadata is already set to the default (0 occupancy)
#             return occupied_gdf, parking_meta_data, empty_gdf # Return valid tuple

#         # --- Aggregate Results for Occupied Parking ---
#         logger.info("Aggregating results for occupied parking...")
#         agg_funcs = {
#             'osmid': 'first', 'area_m2': 'first', 'Category': 'first',
#             'Capacity': 'first', 'geometry': 'first'
#             # Add other relevant columns from gdf_parking if needed
#         }
#         # Check if all agg_funcs keys exist in intersecting_points columns derived from gdf_parking
#         for key in agg_funcs.keys():
#             if key not in intersecting_points.columns and key != 'geometry': # geometry comes from gdf_parking join
#                  logger.warning(f"Column '{key}' needed for aggregation not found in joined data. Check gdf_parking.")
#                  # Handle missing column, e.g., skip aggregation for this key or raise error
#                  # For now, let's remove it from aggregation if missing
#                  if key in agg_funcs: del agg_funcs[key]


#         grouped_occupied = intersecting_points.groupby('index_right').agg(agg_funcs)
#         occupancy_counts = intersecting_points.groupby('index_right').size()
#         grouped_occupied['occupancy'] = occupancy_counts
#         grouped_occupied['occupancy(%)'] = (
#             (grouped_occupied['occupancy'] / grouped_occupied['Capacity'].replace(0, np.nan)) * 100
#         ).fillna(0).round(2)

#         occupied_gdf = gpd.GeoDataFrame(grouped_occupied, geometry='geometry', crs=gdf_parking.crs)
#         occupied_gdf['Capacity'] = occupied_gdf['Capacity'].astype(int)
#         occupied_gdf['area_m2'] = occupied_gdf['area_m2'].round(2)
#         logger.info("Aggregation complete.")

#         # --- Identify Empty Parking ---
#         logger.debug("Identifying empty parking areas...")
#         occupied_parking_indices = set(intersecting_points['index_right'])
#         empty_gdf = gdf_parking[~gdf_parking.index.isin(occupied_parking_indices)].copy()
#         logger.debug(f"Found {len(empty_gdf)} empty parking areas.")

#         # --- Final CRS Conversion ---
#         logger.info(f"Converting output GDFs to target CRS: {crs}")
#         occupied_gdf = occupied_gdf.to_crs(crs)
#         empty_gdf = empty_gdf.to_crs(crs)
#         logger.debug("CRS conversion complete.")

#         # --- Set Geometry and Update Metadata ---
#         occupied_gdf = occupied_gdf.set_geometry("geometry")
#         if not empty_gdf.empty:
#              empty_gdf = empty_gdf.set_geometry("geometry")

#         total_occupied_cars = occupied_gdf['occupancy'].sum()
#         num_occupied_polygons = len(occupied_gdf)
#         num_empty_polygons = len(empty_gdf)
#         occupancy_count_percent = (total_occupied_cars / total_capacity * 100) if total_capacity > 0 else 0
#         estimated_occupied_area = total_occupied_cars * 12.50
#         occupancy_area_percent = (estimated_occupied_area / total_area * 100) if total_area > 0 else 0

#         parking_meta_data.update({
#             "occupancy_per(counts)": round(occupancy_count_percent, 2),
#             "occupancy_per(area)": round(occupancy_area_percent, 2),
#             "Total Occupied (Cars)": int(total_occupied_cars),
#             "parked_area(sq.m)": round(estimated_occupied_area, 2),
#             "Avg Occupancy (Cars)": round(occupied_gdf['occupancy'].mean(), 2) if num_occupied_polygons > 0 else 0.0,
#             "Empty Parking areas": num_empty_polygons,
#             "Occupied Parking areas": num_occupied_polygons
#         })
#         logger.info(f"Intersection analysis complete. Occupied: {num_occupied_polygons}, Empty: {num_empty_polygons}, Cars: {total_occupied_cars}")

#         # --- Select and order columns ---
#         final_occupied_cols = ['osmid', 'area_m2', 'Category', 'Capacity', 'occupancy', 'occupancy(%)', 'geometry']
#         final_empty_cols = ['osmid', 'area_m2', 'Category', 'Capacity', 'geometry']

#         occupied_gdf = occupied_gdf[[col for col in final_occupied_cols if col in occupied_gdf.columns]]
#         empty_gdf = empty_gdf[[col for col in final_empty_cols if col in empty_gdf.columns]]

#         logger.debug("Function returning successfully.")
#         return occupied_gdf, parking_meta_data, empty_gdf # Explicit final return

#     except Exception as e:
#         # --- Catch ANY unexpected error during processing ---
#         logger.exception(f"!!! UNEXPECTED ERROR in intersecting_geojson: {e}") # Use logger.exception to include traceback
#         error_metadata["error"] = f"Internal error during analysis: {e.__class__.__name__}: {e}"
#         # Return the initialized empty GDFs and the metadata containing the error message
#         return occupied_gdf_empty, error_metadata, empty_gdf_empty


def intersecting_geojson(gdf_centroids: gpd.GeoDataFrame, gdf_parking: gpd.GeoDataFrame, bounds: Optional[Tuple[float, float, float, float]], crs="EPSG:4326") -> tuple[gpd.GeoDataFrame, dict[str, Any], gpd.GeoDataFrame]:
    """
    Find intersections between centroids and parking polygons, aggregate data
    (creating MultiPoint geometry for occupied centroids), and identify non-intersecting polygons.
    Uses the user's original core logic.
    """
    logger.info("Starting intersection analysis using original user logic...")
    logger.debug(f"Input centroids CRS: {gdf_centroids.crs}, Input parking CRS: {gdf_parking.crs}, Target CRS: {crs}")

    # --- Initialize error return structure ---
    # Define expected columns early based on aggregation logic + added cols
    occupied_cols = ['osmid', 'area_m2', 'Category', 'Capacity', 'polygon_geometry', 'geometry', 'occupancy', 'occupancy(%)']
    occupied_gdf_empty = gpd.GeoDataFrame(columns=occupied_cols, crs=crs, geometry='geometry') # Set geometry col explicitly
    empty_gdf_cols = gdf_parking.columns.tolist() if 'geometry' in gdf_parking.columns else []
    empty_gdf_empty = gpd.GeoDataFrame(columns=empty_gdf_cols, crs=crs, geometry='geometry' if 'geometry' in empty_gdf_cols else None)

    # Default metadata in case of errors
    error_metadata = {
        "city_name": get_city_name_from_bbox(bounds) if bounds else "Unknown Area",
        "error": "Processing error occurred before analysis completion.",
        "num_parking": 0, "parking_capacity": 0, "parking_area_capacity(sq.m)": 0,
        "total_parking_area(sq.m)": 0, "occupancy_per(counts)": 0.0, "occupancy_per(area)": 0.0,
        "Total Occupied (Cars)": 0, "parked_area(sq.m)": 0.0, "Avg Occupancy (Cars)": 0.0,
        "Empty Parking areas": 0, "Occupied Parking areas": 0
    }

    try:
        # --- Input Validation ---
        if not isinstance(gdf_centroids, gpd.GeoDataFrame) or not isinstance(gdf_parking, gpd.GeoDataFrame):
            error_metadata["error"] = "Invalid input: Must be GeoDataFrames."
            logger.error(error_metadata["error"])
            return occupied_gdf_empty, error_metadata, empty_gdf_empty
        if 'geometry' not in gdf_centroids.columns or 'geometry' not in gdf_parking.columns:
            error_metadata["error"] = "Missing 'geometry' column."
            logger.error(error_metadata["error"])
            return occupied_gdf_empty, error_metadata, empty_gdf_empty
        if 'osmid' not in gdf_parking.columns:
            error_metadata["error"] = "Missing 'osmid' column in parking data."
            logger.error(error_metadata["error"])
            return occupied_gdf_empty, error_metadata, empty_gdf_empty

        # --- Handle Empty Inputs ---
        if gdf_centroids.empty or gdf_parking.empty:
            logger.warning("One or both input GeoDataFrames empty. No intersections possible.")
            error_metadata["error"] = None # No error, just no results
            # Prepare empty/full data for return
            non_intersecting_parking = gdf_parking.copy().to_crs(crs)
            if not non_intersecting_parking.empty: non_intersecting_parking = non_intersecting_parking.set_geometry('geometry')
            # Update metadata counts based on input parking gdf
            error_metadata["num_parking"] = len(gdf_parking['osmid'].unique()) if not gdf_parking.empty else 0
            error_metadata["parking_capacity"] = int(gdf_parking['Capacity'].sum()) if not gdf_parking.empty and 'Capacity' in gdf_parking.columns else 0
            error_metadata["total_parking_area(sq.m)"] = round(gdf_parking['area_m2'].sum(), 2) if not gdf_parking.empty and 'area_m2' in gdf_parking.columns else 0
            error_metadata["Empty Parking areas"] = error_metadata["num_parking"]
            return occupied_gdf_empty, error_metadata, non_intersecting_parking

        # --- CRS Alignment ---
        if gdf_centroids.crs != gdf_parking.crs:
            logger.warning(f"Aligning CRS: centroids {gdf_centroids.crs} -> parking {gdf_parking.crs}")
            gdf_centroids = gdf_centroids.to_crs(gdf_parking.crs)

        # --- Spatial Join (Your Logic) ---
        logger.info("Performing spatial join...")
        # Use 'within' predicate for points strictly inside polygons
        intersecting_points = gpd.sjoin(gdf_centroids, gdf_parking, how='inner', predicate='within') # Changed to 'within'
        logger.info(f"Found {len(intersecting_points)} intersecting points.")

        if intersecting_points.empty:
             logger.warning("No intersections found after spatial join.")
             error_metadata["error"] = None
             non_intersecting_parking = gdf_parking.copy().to_crs(crs)
             if not non_intersecting_parking.empty: non_intersecting_parking = non_intersecting_parking.set_geometry('geometry')
             error_metadata["num_parking"] = len(gdf_parking['osmid'].unique()) if not gdf_parking.empty else 0
             error_metadata["parking_capacity"] = int(gdf_parking['Capacity'].sum()) if not gdf_parking.empty and 'Capacity' in gdf_parking.columns else 0
             error_metadata["total_parking_area(sq.m)"] = round(gdf_parking['area_m2'].sum(), 2) if not gdf_parking.empty and 'area_m2' in gdf_parking.columns else 0
             error_metadata["Empty Parking areas"] = error_metadata["num_parking"]
             return occupied_gdf_empty, error_metadata, non_intersecting_parking


        # --- Merge Polygon Geometry (Your Logic) ---
        # Ensure the index 'index_right' exists and aligns with gdf_parking's index
        if 'index_right' not in intersecting_points.columns:
             error_metadata["error"] = "Spatial join did not produce 'index_right'."
             logger.error(error_metadata["error"])
             return occupied_gdf_empty, error_metadata, empty_gdf_empty

        try:
             # Use .loc for safe index-based access
             intersecting_points['polygon_geometry'] = gdf_parking.loc[intersecting_points['index_right'], 'geometry'].values
        except KeyError as e:
            error_metadata["error"] = f"Error merging polygon geometry. Index mismatch? Missing index in gdf_parking?: {e}"
            logger.error(error_metadata["error"], exc_info=True)
            return occupied_gdf_empty, error_metadata, empty_gdf_empty

        intersecting_points.reset_index(drop=True, inplace=True)

        # --- Aggregation (Your Logic with MultiPoint) ---
        logger.info("Aggregating intersecting points by parking osmid...")
        # Define aggregations, ensure required columns exist in intersecting_points
        required_agg_cols = ['area_m2', 'Category', 'Capacity', 'polygon_geometry', 'geometry']
        if not all(col in intersecting_points.columns for col in required_agg_cols):
             missing = [col for col in required_agg_cols if col not in intersecting_points.columns]
             error_metadata["error"] = f"Missing columns required for aggregation: {missing}"
             logger.error(error_metadata["error"])
             return occupied_gdf_empty, error_metadata, empty_gdf_empty

        aggregations = {
            'area_m2': 'first',
            'Category': 'first',
            'Capacity': 'first',
            "polygon_geometry": 'first',
            'geometry': lambda x: MultiPoint([point for point in x if point is not None and not point.is_empty]) # Combine points, skip invalid
        }
        grouped_df = intersecting_points.groupby('osmid').agg(aggregations).reset_index()

        # Convert back to GeoDataFrame, geometry is now MultiPoint
        # Explicitly set geometry column name during creation
        grouped_gdf = gpd.GeoDataFrame(grouped_df, geometry='geometry', crs=gdf_parking.crs) # Use original CRS before final conversion
        logger.info("Aggregation complete. Grouped GeoDataFrame created.")

        # --- Calculate Occupancy (Your Logic) ---
        grouped_gdf['occupancy'] = intersecting_points.groupby('osmid').size().values # Simpler count using size()
        grouped_gdf['area_m2'] = pd.to_numeric(grouped_gdf['area_m2'], errors='coerce').fillna(0).round(2)
        grouped_gdf['Capacity'] = pd.to_numeric(grouped_gdf['Capacity'], errors='coerce').fillna(0).astype(int)

        # Calculate occupancy percentage, handling potential division by zero
        grouped_gdf['occupancy(%)'] = (
            (grouped_gdf['occupancy'] / grouped_gdf['Capacity'].replace(0, np.nan)) * 100
        ).fillna(0).round(2) # Fill NaN resulting from Capacity=0 with 0%


        # --- Reproject Occupied GDF (Your Logic) ---
        logger.info(f"Converting occupied GDF to target CRS: {crs}")
        parking_gdf = grouped_gdf.to_crs(crs) # This is the final occupied GDF

        # --- Calculate Metadata (Your Logic) ---
        logger.info("Calculating final metadata...")
        total_capacity = gdf_parking['Capacity'].sum()
        total_area = gdf_parking['area_m2'].sum()
        total_occupied_cars = parking_gdf['occupancy'].sum()
        estimated_occupied_area = total_occupied_cars * 12.50

        occupancy_count_percent = (total_occupied_cars / total_capacity * 100) if total_capacity > 0 else 0
        occupancy_area_percent = (estimated_occupied_area / total_area * 100) if total_area > 0 else 0

        # --- Identify Non-Intersecting (Your Logic) ---
        logger.info("Identifying non-intersecting parking polygons...")
        intersecting_parking_indices = set(intersecting_points['index_right']) # Indices from gdf_parking that intersected
        non_intersecting_parking = gdf_parking[~gdf_parking.index.isin(intersecting_parking_indices)].copy()
        non_intersecting_parking = non_intersecting_parking.to_crs(crs) # Convert to target CRS
        if not non_intersecting_parking.empty: non_intersecting_parking = non_intersecting_parking.set_geometry('geometry')
        logger.info(f"Found {len(non_intersecting_parking)} non-intersecting polygons.")

        num_occupied_polygons = len(parking_gdf)
        num_empty_polygons = len(non_intersecting_parking)

        # --- Final Metadata Dictionary (Your Logic) ---
        parking_meta_data = {
            "city_name": get_city_name_from_bbox(bounds) if bounds else "Unknown Area",
            "num_parking": len(gdf_parking['osmid'].unique()),
            "parking_capacity": int(total_capacity),
            "parking_area_capacity(sq.m)": round(total_capacity * 12.50, 2), # Based on total capacity
            "total_parking_area(sq.m)": round(total_area, 2), # Actual total area
            "occupancy_per(counts)": round(occupancy_count_percent, 2),
            "occupancy_per(area)": round(occupancy_area_percent, 2),
            "Total Occupied (Cars)": int(total_occupied_cars),
            "parked_area(sq.m)": round(estimated_occupied_area, 2),
            "Avg Occupancy (Cars)": round(parking_gdf['occupancy'].mean(), 2) if num_occupied_polygons > 0 else 0.0,
            "Empty Parking areas": num_empty_polygons,
            "Occupied Parking areas": num_occupied_polygons,
            "error": None # No error found
        }
        logger.info("Metadata calculation complete.")

        # --- Select and Reorder Columns for Consistency ---
        # Ensure required columns exist before selecting
        final_occupied_cols = ['osmid', 'area_m2', 'Category', 'Capacity', 'polygon_geometry', 'geometry', 'occupancy', 'occupancy(%)']
        final_empty_cols = ['osmid', 'area_m2', 'Category', 'Capacity', 'geometry'] # Assuming these cols exist

        parking_gdf = parking_gdf[[col for col in final_occupied_cols if col in parking_gdf.columns]]
        non_intersecting_parking = non_intersecting_parking[[col for col in final_empty_cols if col in non_intersecting_parking.columns]]

        logger.info("intersecting_geojson function finished successfully.")
        return parking_gdf, parking_meta_data, non_intersecting_parking

    except Exception as e:
        logger.exception(f"!!! UNEXPECTED ERROR in intersecting_geojson: {e}")
        error_metadata["error"] = f"Internal error during analysis: {e.__class__.__name__}: {str(e)}"
        # Return initialized empty GDFs and error metadata
        return occupied_gdf_empty, error_metadata, empty_gdf_empty

def multipoint_to_dataframe(geo_dataframe):
    """
    Converts a GeoDataFrame containing MultiPoint or Point geometries into a pandas DataFrame
    with columns for longitude, latitude, and a weight column indicating the number of points
    in each MultiPoint geometry.
    """
    locations_parking = []
    if 'geometry' not in geo_dataframe.columns: return pd.DataFrame(columns=["longitude", "latitude", "weight"])

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
            locations_parking.append([geom.x, geom.y, 1])

    df_locations_parking = pd.DataFrame(locations_parking, columns=["longitude", "latitude", "weight"])
    return df_locations_parking