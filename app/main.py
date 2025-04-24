# app/main.py
# Main FastAPI application for image detection and parking analysis.

import asyncio
import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, Response
from shapely.geometry.base import BaseGeometry # Import base class for type checking
from shapely.geometry import mapping # Import the mapping function

# --- Local Imports ---
# Make sure models is imported correctly (e.g., from Pydantic models definition)
from app import services, models
# Ensure parking_analysis functions are imported correctly
from backend import parking_analysis
from config import settings # For model loading

logger = logging.getLogger(__name__)

# --- Define Custom JSON Encoder ---
class ShapelyEncoder(json.JSONEncoder):
    """ Custom JSON encoder to handle Shapely geometry objects """
    def default(self, obj):
        if isinstance(obj, BaseGeometry):
            try:
                # Convert Shapely geometry to a GeoJSON-like dictionary structure
                return mapping(obj)
            except Exception as e:
                logger.error(f"Error mapping Shapely object: {e}")
                return str(obj) # Fallback to string representation on error
        # Let the base class default method raise the TypeError for unsupported types
        return super().default(obj)
# --- End Custom Encoder ---

app = FastAPI(title="Image Detection API")

# Global variable to hold the loaded model
model = None

@app.on_event("startup")
async def startup_event():
    """ Load the object detection model during application startup. """
    global model
    # Assuming settings.model handles the actual loading
    model = settings.model
    if model is None:
        logger.critical("Model failed to load during startup. Application cannot proceed.")
        # Depending on deployment, you might want the app to fail hard here.
        # For now, raise RuntimeError which will stop FastAPI startup.
        raise RuntimeError("Model failed to load during startup.")
    logger.info("Object detection model loaded successfully.")

@app.get("/", include_in_schema=False)
async def root_redirect():
    """ Redirect root path to API documentation. """
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """ Provide a favicon to prevent 404 errors from browsers. """
    # Return No Content response
    return Response(status_code=204)

@app.post("/detect_and_analyze_parking/", response_model=models.ParkingAnalysisOutput)
async def detect_and_analyze_parking(file: UploadFile = File(...)):
    """
    Receives an image file, performs object detection (cars), extracts metadata,
    analyzes parking space occupancy using OSM data, and returns results
    including occupied/empty parking GeoJSON and image metadata.

    - Decodes the image.
    - Runs object detection inference.
    - Extracts geographic bounding box from image metadata.
    - Creates GeoDataFrame of detected object centroids.
    - Fetches parking polygons from OpenStreetMap for the image area.
    - Performs spatial intersection to determine occupied/empty parking.
    - Returns structured GeoJSON results.
    """
    logger.info(f"Received request for detection and parking analysis for file: {file.filename}")
    occupied_gdf = None
    empty_gdf = None
    image_bbox_list = None # Expected format: [top_lat, min_lon, bottom_lat, max_lon] (EPSG:4326)
    gdf_centroids = None

    try:
        # 1. Read File Contents (Async I/O)
        contents = await file.read()
        if not contents:
            logger.warning("Received empty file.")
            raise HTTPException(status_code=400, detail="Empty file received.")
        logger.debug("File read successfully.")

        # 2. Decode Image (Run blocking cv2 in thread pool)
        def decode_image(content_bytes):
            """ Safely decodes image bytes using OpenCV. """
            nparr = np.frombuffer(content_bytes, np.uint8)
            img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_decoded is None:
                # Log the specific reason if possible (e.g., file format)
                logger.error("Failed to decode image. Check file format and integrity.")
                raise ValueError("Could not decode image. Ensure it is a valid image format (e.g., JPEG, PNG).")
            return img_decoded

        img = await asyncio.to_thread(decode_image, contents)
        logger.info("Image decoded successfully.")

        # 3. Run Object Detection (Run blocking inference in thread pool)
        if model is None:
             logger.error("Model is not loaded. Cannot perform inference.")
             raise HTTPException(status_code=503, detail="Model not available. Service may be starting up or encountered an error.")

        detections = await asyncio.to_thread(services.run_inference, model, img)
        logger.info(f"Object detection complete. Found {len(detections)} potential objects.")

        # 4. Create Centroid GeoDataFrame and Get Image BBox List (Run blocking service in thread)
        # This call now correctly passes 3 arguments as defined in services.py
        geo_data = await asyncio.to_thread(services.create_geodataframe, contents, detections, img)

        # Extract results from the dictionary returned by create_geodataframe
        gdf_centroids = geo_data.get("gdf")
        image_bbox_list = geo_data.get("image_bbox") # Expected: [top_lat, min_lon, bottom_lat, max_lon]

        # --- Validate results from create_geodataframe ---
        if image_bbox_list is None:
            logger.error("Image bounding box (image_bbox) could not be determined from metadata.")
            # This is often due to missing EXIF/XMP tags or incorrect metadata format
            raise HTTPException(status_code=400, detail="Could not determine geographic bounding box from image metadata.")
        if not isinstance(image_bbox_list, list) or len(image_bbox_list) != 4:
             logger.error(f"Invalid format for image bounding box: {image_bbox_list}")
             raise HTTPException(status_code=500, detail="Internal error: Invalid image bounding box format generated.")

        if gdf_centroids is None: # Allow empty GDF, but not None
             logger.error("Centroid GeoDataFrame (gdf) creation failed unexpectedly (returned None).")
             raise HTTPException(status_code=500, detail="Internal server error: Failed to create centroid data.")
        if gdf_centroids.empty:
            logger.warning("No centroids generated from detections, possibly no relevant objects detected or issues in georeferencing.")
            # Proceeding, but analysis results will likely be empty.

        logger.info(f"Centroid GDF created with {len(gdf_centroids)} points. Image bbox (TopLat, MinLon, BotLat, MaxLon): {image_bbox_list}")

        # 5. Convert Centroids GDF to GeoJSON dict (Run blocking conversion in thread)
        def gdf_to_json_dict_sync(gdf):
             """ Converts GeoDataFrame to GeoJSON dictionary using standard method. """
             if gdf is None or gdf.empty:
                  return {"type": "FeatureCollection", "features": []}
             # Standard to_json is usually sufficient for Point geometries from centroids
             return json.loads(gdf.to_json())

        centroids_geojson_dict = await asyncio.to_thread(gdf_to_json_dict_sync, gdf_centroids)

        # Structure the image metadata part of the response
        image_metadata_dict = {
            "image_bbox": image_bbox_list, # [top_lat, min_lon, bottom_lat, max_lon]
            "image_bbox_centers": centroids_geojson_dict # GeoJSON FeatureCollection of points
        }

        # 6. Get Parking Data from OSM (Run blocking I/O and processing in thread)
        logger.info("Fetching parking data from OSM...")

        # Validate lat/lon ranges before passing to OSM query
        # bbox format: [top_lat, min_lon, bottom_lat, max_lon]
        if not (-90 <= image_bbox_list[0] <= 90 and -180 <= image_bbox_list[1] <= 180 and \
                -90 <= image_bbox_list[2] <= 90 and -180 <= image_bbox_list[3] <= 180):
            logger.error(f"Invalid latitude/longitude values in image bounding box: {image_bbox_list}")
            raise HTTPException(status_code=400, detail="Invalid geographic coordinates derived from image metadata. Check latitude (-90 to 90) and longitude (-180 to 180).")

        # Prepare bounds for get_parking_data (expects left, bottom, right, top which is min_lon, bottom_lat, max_lon, top_lat)
        analysis_bounds = (image_bbox_list[1], image_bbox_list[2], image_bbox_list[3], image_bbox_list[0])

        # Wrap parking_analysis.get_parking_data in asyncio.to_thread
        parking_poly_df = await asyncio.to_thread(parking_analysis.get_parking_data, analysis_bounds)

        # --- Handle Case: No Parking Polygons Found ---
        if parking_poly_df is None or parking_poly_df.empty:
            logger.warning(f"No parking polygons found via OSM for the area defined by bounds: {analysis_bounds}")

            # Try to get city name for context even if no parking found
            city_name = "Unknown Area"
            try:
                 # Prepare bounds for get_city_name_from_bbox (expects min_lat, min_lon, max_lat, max_lon)
                 city_lookup_bounds = (image_bbox_list[2], image_bbox_list[1], image_bbox_list[0], image_bbox_list[3])
                 city_name = await asyncio.to_thread(parking_analysis.get_city_name_from_bbox, city_lookup_bounds)
                 logger.info(f"Determined city name as: {city_name}")
            except Exception as city_err:
                 logger.warning(f"Could not determine city name from bbox {city_lookup_bounds}: {city_err}")

            # Return response indicating no parking found, but include image metadata
            return models.ParkingAnalysisOutput(
                analysis_metadata={
                    "message": "No parking polygons found in the specified area.",
                    "city_name": city_name,
                    "error": None,
                    "Total Occupied (Cars)": 0 # No parking = 0 occupied
                 },
                occupied_parking={"type": "FeatureCollection", "features": []},
                empty_parking={"type": "FeatureCollection", "features": []},
                image_metadata=image_metadata_dict
            )
        # --- End Handle Case ---

        logger.info(f"Retrieved {len(parking_poly_df)} parking polygons from OSM.")

        # 7. Perform Intersection Analysis (Run blocking geo-processing in thread)
        logger.info("Performing intersection analysis between detected objects and parking polygons...")
        # Prepare bounds for intersecting_geojson's internal city lookup
        # (expects min_lat, min_lon, max_lat, max_lon)
        intersection_bounds = (image_bbox_list[2], image_bbox_list[1], image_bbox_list[0], image_bbox_list[3])

        # Wrap parking_analysis.intersecting_geojson in asyncio.to_thread
        analysis_result = await asyncio.to_thread(
            parking_analysis.intersecting_geojson,
            gdf_centroids, parking_poly_df, intersection_bounds
        )

        # Validate the structure of the analysis result
        if not isinstance(analysis_result, tuple) or len(analysis_result) != 3:
            logger.error(f"Intersecting_geojson returned unexpected type or structure: {type(analysis_result)}")
            raise HTTPException(status_code=500, detail="Internal error: Unexpected analysis result format received.")

        occupied_gdf, parking_metadata, empty_gdf = analysis_result
        logger.info("Intersection analysis complete.")

        # Check for errors reported *within* the analysis metadata
        if parking_metadata and parking_metadata.get("error"):
            error_message = parking_metadata['error']
            logger.error(f"Error reported from intersection analysis function: {error_message}")
            # Return the error within the response structure
            # Ensure parking_metadata has expected keys even on error for the model
            parking_metadata.setdefault("city_name", "Unknown")
            parking_metadata.setdefault("Total Occupied (Cars)", 0) # Default if error occurred before counting

            return models.ParkingAnalysisOutput(
                  analysis_metadata=parking_metadata, # Include the error message here
                  occupied_parking={"type": "FeatureCollection", "features": []}, # Return empty GeoJSON on error
                  empty_parking={"type": "FeatureCollection", "features": []},
                  image_metadata=image_metadata_dict)
            # Alternative: raise HTTPException(status_code=500, detail=f"Analysis error: {error_message}")

        # Log analysis summary
        occupied_count = len(occupied_gdf) if occupied_gdf is not None else 0
        empty_count = len(empty_gdf) if empty_gdf is not None else 0
        logger.info(f"Analysis Results: Occupied spots={occupied_count}, Empty spots={empty_count}, City={parking_metadata.get('city_name', 'N/A')}")


        # 8. Convert Result GDFs to GeoJSON Dictionaries (Run blocking conversion in thread)
        # Use the custom ShapelyEncoder for potentially complex geometries (polygons)
        def gdf_to_json_dict_sync_shapely(gdf):
            """ Converts GeoDataFrame to GeoJSON dict using ShapelyEncoder for complex geometries. """
            if gdf is None or gdf.empty:
                return {"type": "FeatureCollection", "features": []}
            # Use dumps with cls=ShapelyEncoder, then loads to get dict
            return json.loads(gdf.to_json(cls=ShapelyEncoder)) # Pass encoder directly to to_json if supported, otherwise use dumps/loads
            # Note: gdf.to_json() might not directly accept `cls`. If it fails, use:
            # return json.loads(json.dumps(json.loads(gdf.to_json()), cls=ShapelyEncoder)) # More convoluted but works

        occupied_geojson_dict = await asyncio.to_thread(gdf_to_json_dict_sync_shapely, occupied_gdf)
        empty_geojson_dict = await asyncio.to_thread(gdf_to_json_dict_sync_shapely, empty_gdf)

        # 9. Structure and Return Final Response
        final_response = models.ParkingAnalysisOutput(
            analysis_metadata=parking_metadata,
            occupied_parking=occupied_geojson_dict,
            empty_parking=empty_geojson_dict,
            image_metadata=image_metadata_dict
        )
        logger.info("Analysis successful. Returning results.")
        return final_response

    # --- Exception Handling ---
    except FileNotFoundError as fnfe:
        # Specific case if an internal file path is wrong (less likely for UploadFile)
        logger.error(f"File not found error during processing: {fnfe}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: Required file not found - {fnfe}")
    except ValueError as ve:
        # Catches issues from decode_image, metadata validation, GDF creation value errors, etc.
        logger.warning(f"Data validation or processing error: {ve}", exc_info=True) # Log traceback for debugging
        # Return 400 for bad input data or unmet value requirements
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        # Catches issues from model inference, geo-processing errors wrapped in services/analysis
        logger.error(f"Runtime error during processing: {re}", exc_info=True)
        # Return 500 for internal processing errors
        raise HTTPException(status_code=500, detail=str(re))
    except KeyError as ke:
        # Specifically for missing keys, often in metadata
        logger.error(f"Missing key error, likely in processing data structures or metadata: {ke}", exc_info=True)
        # Often indicates bad input metadata or unexpected structure
        raise HTTPException(status_code=400, detail=f"Missing required data key: {ke}")
    except ImportError as ie:
        # Errors related to missing dependencies
        logger.error(f"Import error: {ie}. Check dependencies and paths.", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server configuration error: Missing dependency - {ie}")
    except HTTPException as he:
        # Re-raise HTTPExceptions that were raised intentionally within the try block
        logger.warning(f"HTTP Exception raised during processing: Status={he.status_code}, Detail={he.detail}")
        raise he
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred: {e.__class__.__name__}: {e}", exc_info=True)
        # Return a generic 500 error for unknown issues
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e.__class__.__name__}")


if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    # Use --reload for development only, remove for production deployments (e.g., Docker)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) # Add reload=True for dev
    # Production command in Dockerfile/supervisor would typically be without reload:
    # uvicorn app.main:app --host 0.0.0.0 --port 8000