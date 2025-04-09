import cv2
import os
import numpy as np
import supervision as sv  # Assuming sv is a custom detection library
from typing import List, Dict

#from ultralytics import YOLO 

def process_image_slice(image_slice: np.ndarray, model) -> sv.Detections:
    """
    Processes a slice of an image and returns detections.

    Args:
        image_slice (np.ndarray): A slice of the input image.
        model: The pre-trained detection model to use.

    Returns:
        sv.Detections: Detections object containing the model's results.
    """
    try:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {e}")

def create_inference_slicer(model, image_size: tuple = (640,640), overlap: tuple = (0.0, 0.0), iou_threshold: float = 0.5):
    """
    Creates an inference slicer for processing images.

    Args:
        model: Pre-trained model for detection.
        image_size (tuple): Width and height of slices.
        overlap (tuple): Overlap ratio between slices.
        iou_threshold (float): IOU threshold for detections.

    Returns:
        sv.InferenceSlicer: Configured slicer object.
    """
    return sv.InferenceSlicer(
        callback=lambda x: process_image_slice(x, model),
        slice_wh=image_size,
        overlap_ratio_wh=overlap,
        iou_threshold=iou_threshold
    )

def convert_detections_to_list(detections: sv.Detections) -> List[Dict[str, any]]:
    """
    Converts detections into a list of dictionaries with bounding boxes, class names, and confidence scores.

    Args:
        detections (sv.Detections): Detections object containing the model's output.

    Returns:
        List[Dict]: A list of dictionaries, each containing 'bbox', 'class', and 'confidence'.
    """
    try:
        xyxy_array = detections.xyxy
        confidence_array = detections.confidence
        class_names = detections.data.get('class_name', [])

        if len(confidence_array) != len(class_names):
            raise ValueError("Mismatched lengths between confidence scores and class names.")

        # Construct list of detections with reordered bounding box format
        return [
            {
                "bbox": xyxy_array[i].tolist(),  # Reordered bbox
                "class": class_names[i],
                "confidence": float(confidence_array[i])
            }
            for i in range(len(confidence_array))
        ]

    except Exception as e:
        raise RuntimeError(f"Error converting detections to list: {e}") 
    
def save_predicted_image(image: np.ndarray, filename: str, detections: sv.Detections):
    output_path = os.path.join("predictions", filename)
    os.makedirs("predictions", exist_ok=True)

    # Use supervision BoxAnnotator for drawing bounding boxes
    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=image, detections=detections)

    cv2.imwrite(output_path, annotated_image)
    print(f"Saved: {output_path}")

def save_predicted_image(image: np.ndarray, img_path: str, detections: sv.Detections):
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
    

def run_inference(model, img_path: str) -> List[Dict[str, any]]:
    """
    Runs inference on an image file using the provided model and converts the detections to a list.

    Args:
        model: Pre-trained detection model.
        img_path (str): Path to the input image.

    Returns:
        List[Dict]: A list of dictionaries containing detection results.
    """
    try:
        image_raw = cv2.imread(img_path)  # Load image from file

        if image_raw is None:
            raise ValueError(f"Could not load image from path: {img_path}")

        slicer = create_inference_slicer(model)
        detections = slicer(image_raw)

        # Save the predicted image
        save_predicted_image(image_raw, img_path, detections)

        return convert_detections_to_list(detections)
    except Exception as e:
        raise RuntimeError(f"Error during image inference: {e}")
