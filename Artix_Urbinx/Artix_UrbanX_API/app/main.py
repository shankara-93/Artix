from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, Response
from typing import List, Dict, Any
# --- Make sure models is imported correctly ---
from app import services, models
# --- Ensure parking_analysis is imported correctly ---
from backend import parking_analysis
from Artix_Urbinx_API.config import settings
import uvicorn
import cv2
import numpy as np
import json # Needed for json.loads
import logging
import geopandas as gpd # Keep if needed elsewhere, but maybe not directly here
from shapely.geometry.base import BaseGeometry # Import base class for type checking
from shapely.geometry import mapping # Import the mapping function

logger = logging.getLogger(__name__)




# --- Define Custom JSON Encoder ---
class ShapelyEncoder(json.JSONEncoder):
    """ Custom JSON encoder to handle Shapely geometry objects """
    def default(self, obj):
        # Check if the object is a known Shapely geometry type
        if isinstance(obj, BaseGeometry):
            try:
                # Convert Shapely geometry to GeoJSON-like dictionary
                return mapping(obj)
            except Exception as e:
                logger.error(f"Error mapping Shapely object: {e}")
                # Fallback or raise error? Returning string representation for now
                return str(obj)
        # Let the base class default method raise the TypeError for other types
        return super().default(obj)
# --- End Custom Encoder ---

app = FastAPI(title="Image Detection API")

@app.on_event("startup")
# ... (startup remains the same) ...
async def startup_event():
    global model
    model = settings.model
    if model is None:
        raise RuntimeError("Model failed to load during startup.")

@app.get("/", include_in_schema=False)
# ... (root_redirect remains the same) ...
async def root_redirect():
    return RedirectResponse(url="/docs")

# --- No changes needed for favicon endpoint ---
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/detect_and_analyze_parking/", response_model=models.ParkingAnalysisOutput)
async def detect_and_analyze_parking(file: UploadFile = File(...)):
    """
    Detects objects, analyzes parking, returns results including image metadata
    (bbox and detected centroids GeoJSON).
    """
    logger.info("Received request for detection and parking analysis.")
    occupied_gdf = None
    empty_gdf = None
    image_bbox_list = None
    gdf_centroids = None
    try:
        # 1. Read and Decode Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Could not decode image.")
        logger.info("Image decoded.")

        # 2. Run Object Detection
        detections = services.run_inference(model, img)
        logger.info(f"Object detection complete. Found {len(detections)} potential objects.")

        # 3. Create Centroid GeoDataFrame and Get Image BBox List
        geo_data = services.create_geodataframe(contents, detections)
        gdf_centroids = geo_data["gdf"]
        image_bbox_list = geo_data["image_bbox"]
        logger.info(f"Centroid GDF created with {len(gdf_centroids)} points. Image bbox list: {image_bbox_list}")

        if image_bbox_list is None:
             logger.error("Image bounding box list could not be determined.")
             raise HTTPException(status_code=500, detail="Internal server error: Could not determine image bounding box.")
        if gdf_centroids is None:
             logger.error("Centroid GeoDataFrame could not be created.")
             raise HTTPException(status_code=500, detail="Internal server error: Could not create centroid data.")

        # --- Convert Centroids GDF to GeoJSON dict (using encoder) ---
        centroids_geojson_dict = json.loads(gdf_centroids.to_json())
        # --- End Conversion ---

        # --- Create the image metadata dictionary ---
        # Rename variable for consistency with model
        image_metadata_dict = {
            "image_bbox": image_bbox_list,
            "image_bbox_centers": centroids_geojson_dict
        }
        # --- End Change ---

        # 4. Get Parking Data from OSM
        logger.info("Fetching parking data from OSM...")
        analysis_bounds = (image_bbox_list[2], image_bbox_list[1], image_bbox_list[0], image_bbox_list[3])
        parking_poly_df = parking_analysis.get_parking_data(analysis_bounds)

        if parking_poly_df is None or parking_poly_df.empty:
             logger.warning("No parking polygons found for the given image area.")
             city_name = parking_analysis.get_city_name_from_bbox(analysis_bounds) if analysis_bounds else "Unknown Area"
             # --- FIX early return assignment ---
             return models.ParkingAnalysisOutput(
                  analysis_metadata={"message": "No parking polygons found.", "city_name": city_name, "error": None},
                  occupied_parking={"type": "FeatureCollection", "features": []},
                  empty_parking={"type": "FeatureCollection", "features": []},
                  image_metadata=image_metadata_dict # <-- Use correct keyword 'image_metadata'
             )
             # --- End FIX ---
        logger.info(f"Retrieved {len(parking_poly_df)} parking polygons.")

        # 5. Perform Intersection Analysis
        logger.info("Performing intersection analysis...")
        analysis_result = parking_analysis.intersecting_geojson(
            gdf_centroids, parking_poly_df, analysis_bounds
        )
        if not isinstance(analysis_result, tuple) or len(analysis_result) != 3:
             logger.error(f"Intersecting_geojson returned unexpected type: {type(analysis_result)}")
             raise HTTPException(status_code=500, detail="Internal error: Unexpected analysis result format.")

        occupied_gdf, parking_metadata, empty_gdf = analysis_result
        logger.info("Intersection analysis complete.")

        if parking_metadata.get("error"):
            logger.error(f"Error reported from intersection analysis: {parking_metadata['error']}")
            raise HTTPException(status_code=500, detail=f"Analysis error: {parking_metadata['error']}")

        # Convert GDFs to GeoJSON dicts USING CUSTOM ENCODER
        occupied_geojson_dict = json.loads(occupied_gdf.to_json(cls=ShapelyEncoder))
        empty_geojson_dict = json.loads(empty_gdf.to_json(cls=ShapelyEncoder))

        # 6. Structure and Return Final Response
        # --- FIX final response assignment ---
        final_response = models.ParkingAnalysisOutput(
            analysis_metadata=parking_metadata,
            occupied_parking=occupied_geojson_dict,
            empty_parking=empty_geojson_dict,
            image_metadata=image_metadata_dict # <-- Use correct keyword 'image_metadata'
        )
        # --- End FIX ---
        logger.info("Analysis successful. Returning results.")
        return final_response

    # --- Exception Handling remains the same ---
    except ValueError as ve:
         logger.error(f"Value error during processing: {ve}", exc_info=True)
         raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
         logger.error(f"RuntimeError during processing: {re}", exc_info=True)
         raise HTTPException(status_code=500, detail=str(re))
    except KeyError as ke:
        logger.error(f"Missing key error: {ke}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Missing required data key: {ke}")
    except ImportError as ie:
        logger.error(f"Import error: {ie}. Check dependencies and paths.", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server setup error: {ie}")
    except Exception as e:
        logger.error(f"Unexpected error: {e.__class__.__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e.__class__.__name__}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)