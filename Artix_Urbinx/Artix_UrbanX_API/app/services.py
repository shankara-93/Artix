# app/services.py
import cv2
import geopandas as gpd 
import numpy as np
import supervision as sv
from typing import List, Dict
import logging
from backend.image_metadata import load_jpg_metadata, CoordinateTransformer
from backend.pixeltogeojson import boundingboxestogeodataframe
import os  # Import the 'os' module

logger = logging.getLogger(__name__)


def process_image_slice(image_slice: np.ndarray, model) -> sv.Detections:
    """Processes a slice of an image and returns detections."""
    try:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)
    except Exception as e:
        logger.exception("Error during model inference")
        raise RuntimeError(f"Error during model inference: {e}")


def create_inference_slicer(model, image_size: tuple = (640, 640), overlap: tuple = (0.0, 0.0), iou_threshold: float = 0.5):
    """Creates an inference slicer for processing images."""
    return sv.InferenceSlicer(
        callback=lambda x: process_image_slice(x, model),
        slice_wh=image_size,
        overlap_ratio_wh=overlap,
        iou_threshold=iou_threshold
    )


def convert_detections_to_list(detections: sv.Detections) -> List[Dict[str, any]]:
    """Converts detections into a list of dictionaries."""
    try:
        xyxy_array = detections.xyxy
        confidence_array = detections.confidence
        class_ids = detections.class_id  # Using class_id, assuming this is correct
        class_names = detections.data.get('class_name', [])  # Get class names

        if len(confidence_array) != len(class_ids):
            raise ValueError("Mismatched lengths between confidence scores and class IDs.")

        return [
            {
                "bbox": xyxy_array[i].tolist(),
                "class": class_names[i],  # Access class name directly
                "confidence": float(confidence_array[i])
            }
            for i in range(len(confidence_array))
        ]

    except Exception as e:
        logger.exception("Error converting detections to list")
        raise RuntimeError(f"Error converting detections to list: {e}")


def save_predicted_image(image: np.ndarray, img_path: str, detections: sv.Detections):
    """Saves the predicted image with bounding boxes."""
    try:
        # Extract folder name and filename
        folder_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.basename(img_path)
        save_folder = os.path.join("predictions", folder_name)
        os.makedirs(save_folder, exist_ok=True)

        # Construct output filename
        output_path = os.path.join(save_folder, "predictions_" + file_name)

        # Check if file already exists
        if os.path.exists(output_path):
            print(f"Skipped saving: {output_path} (already exists)")
            return

        # Use supervision BoxAnnotator for drawing bounding boxes
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)

        cv2.imwrite(output_path, annotated_image)
        print(f"Saved: {output_path}")

    except Exception as e:
        logger.exception("Error saving predicted image")
        print(f"Error saving predicted image: {e}")


def run_inference(model, image: np.ndarray) -> List[Dict[str, any]]:  # Corrected return type
    """Runs inference on an image and returns detections."""
    try:
        slicer = create_inference_slicer(model)
        detections = slicer(image)
        return convert_detections_to_list(detections)  # Returning detection result list
    except Exception as e:
        logger.exception("Error during image inference")
        raise RuntimeError(f"Error during image inference: {e}")


def create_geodataframe(image_bytes: bytes, detections: List[Dict[str, any]]) -> gpd.GeoDataFrame:
    """Creates a GeoDataFrame from detections and image metadata."""
    try:
        metadata = load_jpg_metadata(image_bytes)  # pass in the image_bytes now
        if not metadata:
            raise ValueError("Could not load metadata from image.")

        transformer = CoordinateTransformer()

        # Check if metadata['ImageDescription'] exists, if not try 'UserComment'
        metadata_source = metadata.get('ImageDescription') or metadata.get('UserComment')
        if not metadata_source:
             raise ValueError("Required metadata 'ImageDescription' or 'UserComment' not found in image.")

        bbox_info = transformer.calculate_bbox_and_center(
            metadata_source['top_left_x'],
            metadata_source['top_left_y'],
            metadata_source['width_px'],
            metadata_source['height_px'],
            metadata_source['pixel_size'],
            from_crs=metadata_source['crs'],
            to_crs="EPSG:4326"
        )
        # This section needs to load image and get image dimensions to run detections.
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # Load it for getting dimensions
        image_height, image_width = image.shape[:2]
        gdf = boundingboxestogeodataframe(detections, image_width, image_height,
                                          bbox_info['bounding_box'])
        return {"gdf": gdf,"image_bbox": bbox_info['bounding_box'] }
    except Exception as e:
        logger.exception("Error creating GeoDataFrame")
        raise RuntimeError(f"Error creating GeoDataFrame: {e}")