# app/services.py
import cv2
import geopandas as gpd
import numpy as np
import supervision as sv
from typing import List, Dict, Optional,Any# Added Optional
import logging
# Make sure these are imported if needed within the function now
from backend.image_metadata import load_jpg_metadata, CoordinateTransformer
from backend.pixeltogeojson import boundingboxestogeodataframe
import os

logger = logging.getLogger(__name__)


def process_image_slice(image_slice: np.ndarray, model) -> sv.Detections:
    """Processes a slice of an image and returns detections."""
    try:
        result = model(image_slice)[0]
        # It's generally better to return raw results and add names later if needed
        return sv.Detections.from_ultralytics(result)
    except Exception as e:
        logger.exception("Error during model inference")
        # Don't raise generic RuntimeError, let specific exceptions propagate or wrap if needed
        raise # Re-raise the original exception for better debugging


def create_inference_slicer(model, image_size: tuple = (640, 640), overlap: tuple = (0.0, 0.0), iou_threshold: float = 0.5):
    """Creates an inference slicer for processing images."""
    # Add error handling for model loading or invalid parameters if necessary
    return sv.InferenceSlicer(
        callback=lambda x: process_image_slice(x, model),
        slice_wh=image_size,
        overlap_ratio_wh=overlap,
        iou_threshold=iou_threshold
    )


def convert_detections_to_list(detections: sv.Detections, class_names_map: Optional[Dict[int, str]] = None) -> List[Dict[str, any]]:
    """
    Converts detections into a list of dictionaries.
    Optionally uses a class ID to name map.
    """
    detection_list = []
    try:
        xyxy_array = detections.xyxy
        confidence_array = detections.confidence
        class_ids = detections.class_id
        # Try getting names from detections.data first (might be populated by custom models)
        # Fall back to using the provided map or generating default names
        class_names_in_data = detections.data.get('class_name')

        num_detections = len(confidence_array) if confidence_array is not None else 0

        # Validate lengths
        if not (len(xyxy_array) == num_detections and len(class_ids) == num_detections):
             raise ValueError("Mismatched lengths between detection attributes (xyxy, confidence, class_id).")
        # Validate class_names_in_data if present
        if class_names_in_data is not None and len(class_names_in_data) != num_detections:
             logger.warning("Length of 'class_name' in detections.data does not match number of detections. Ignoring.")
             class_names_in_data = None

        for i in range(num_detections):
            class_id = class_ids[i]
            class_name = None
            if class_names_in_data is not None: # Explicitly check for None
                 class_name = class_names_in_data[i]
            elif class_names_map:
                 class_name = class_names_map.get(class_id, f"class_{class_id}") # Use map, fallback to default
            else:
                 class_name = f"class_{class_id}" # Default name if no map provided

            detection_list.append({
                "bbox": xyxy_array[i].tolist(),
                "class": class_name, # Use determined class name
                "confidence": float(confidence_array[i])
            })
        return detection_list

    except Exception as e:
        logger.exception("Error converting detections to list")
        # Avoid raising generic RuntimeError, let specific exceptions propagate or wrap
        raise ValueError(f"Error converting detections to list: {e}")


def save_predicted_image(image: np.ndarray, img_path: str, detections: sv.Detections):
    """Saves the predicted image with bounding boxes."""
    # (Keep this function as is if needed elsewhere)
    try:
        folder_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.basename(img_path)
        save_folder = os.path.join("predictions", folder_name)
        os.makedirs(save_folder, exist_ok=True)
        output_path = os.path.join(save_folder, "predictions_" + file_name)
        if os.path.exists(output_path):
            print(f"Skipped saving: {output_path} (already exists)")
            return
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved: {output_path}")
    except Exception as e:
        logger.exception("Error saving predicted image")
        print(f"Error saving predicted image: {e}")


def run_inference(model, image: np.ndarray) -> List[Dict[str, any]]:
    """Runs inference on an image and returns detections as a list of dicts."""
    try:
        slicer = create_inference_slicer(model)
        detections = slicer(image) # Detections object

        # Get class ID to name mapping from the model if available
        class_names_map = None
        if hasattr(model, 'names') and isinstance(model.names, dict):
             class_names_map = model.names
        elif hasattr(model, 'model') and hasattr(model.model, 'names') and isinstance(model.model.names, dict):
             # Sometimes names are nested under model.model
             class_names_map = model.model.names

        # Convert detections object to list of dicts, passing the name map
        return convert_detections_to_list(detections, class_names_map)
    except Exception as e:
        logger.exception("Error during image inference or detection conversion")
        # Let specific errors propagate or wrap them
        raise RuntimeError(f"Error during image inference: {e}")


# --- MODIFIED FUNCTION SIGNATURE ---
def create_geodataframe(image_bytes: bytes, detections: List[Dict[str, any]], image: np.ndarray) -> Dict[str, Any]:
    """
    Creates a GeoDataFrame from detections and image metadata.
    Accepts the already decoded image array. Returns dict with GDF and bbox.
    """
    try:
        # Load metadata from bytes
        metadata = load_jpg_metadata(image_bytes) # This func logs errors internally
        if not metadata:
            # Log here as well, as load_jpg_metadata might return {} silently
            logger.error("Failed to load any metadata from image bytes.")
            raise ValueError("Could not load metadata from image.")

        transformer = CoordinateTransformer()

        # Prioritize 'ImageDescription', then 'UserComment'
        metadata_source = metadata.get('ImageDescription') or metadata.get('UserComment')
        if not metadata_source or not isinstance(metadata_source, dict): # Check it's a dict too
             logger.error(f"Required metadata dict not found in 'ImageDescription' or 'UserComment'. Found: {metadata_source}")
             raise ValueError("Required metadata 'ImageDescription' or 'UserComment' (as dict) not found in image.")

        # Ensure all required keys are in metadata_source
        required_keys = ['top_left_x', 'top_left_y', 'width_px', 'height_px', 'pixel_size', 'crs']
        if not all(key in metadata_source for key in required_keys):
             missing_keys = [key for key in required_keys if key not in metadata_source]
             logger.error(f"Missing required metadata keys: {missing_keys}. Metadata found: {metadata_source}")
             raise ValueError(f"Missing required metadata keys: {missing_keys}")

        # Add type checks for metadata values if possible/needed
        # e.g., ensure pixel_size, coordinates are numbers

        bbox_info = transformer.calculate_bbox_and_center(
            metadata_source['top_left_x'],
            metadata_source['top_left_y'],
            metadata_source['width_px'],
            metadata_source['height_px'],
            metadata_source['pixel_size'],
            from_crs=metadata_source['crs'],
            to_crs="EPSG:4326" # Target CRS for output bbox
        )

        # --- USE PASSED IMAGE FOR DIMENSIONS ---
        if image is None or not isinstance(image, np.ndarray) or image.ndim < 2:
             logger.error("Invalid image array passed to create_geodataframe.")
             raise ValueError("Invalid image array provided.")
        image_height, image_width = image.shape[:2] # Use dimensions from passed image array
        # --- END CHANGE ---

        # Ensure detections is a list of dicts as expected by boundingboxestogeodataframe
        if not isinstance(detections, list):
             logger.error(f"Detections passed to create_geodataframe is not a list: {type(detections)}")
             raise ValueError("Invalid format for detections.")
        # Optionally add checks for dict structure within the list

        # Create the GeoDataFrame
        gdf = boundingboxestogeodataframe(detections, image_width, image_height,
                                          bbox_info['bounding_box']) # Uses EPSG:4326 bounding box

        # Check if GDF creation was successful
        if gdf is None or gdf.empty:
             logger.warning("boundingboxestogeodataframe returned None or empty GeoDataFrame.")
             # Return empty GDF in the result dict to avoid downstream errors
             gdf = gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs="EPSG:4326")


        return {"gdf": gdf, "image_bbox": bbox_info['bounding_box']}

    except ValueError as ve: # Catch specific expected errors
         logger.error(f"Value error creating GeoDataFrame: {ve}", exc_info=True) # Log traceback for value errors too
         raise # Re-raise to be caught by main endpoint handler
    except KeyError as ke:
         logger.error(f"Key error accessing metadata: {ke}. Metadata: {metadata_source}", exc_info=True)
         raise ValueError(f"Missing key in metadata: {ke}") # Raise ValueError for consistency
    except Exception as e:
        logger.exception("Unexpected error creating GeoDataFrame") # Log full traceback for unexpected
        raise RuntimeError(f"Error creating GeoDataFrame: {e}") # Wrap unexpected errors