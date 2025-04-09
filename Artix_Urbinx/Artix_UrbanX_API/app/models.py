from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional 

class ParkingAnalysisOutput(BaseModel):
    analysis_metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata containing analysis results like counts, percentages, city name."
    )
    occupied_parking: Dict[str, Any] = Field(
        ...,
        description="GeoJSON FeatureCollection dict of points representing occupied parking areas."
    )
    empty_parking: Dict[str, Any] = Field(
        ...,
        description="GeoJSON FeatureCollection dict of polygons representing empty parking areas."
    )
    # --- FIX: Rename this field to match main.py ---
    image_metadata: Optional[Dict[str, Any]] = Field( # <-- RENAMED to image_metadata
        None, # Default value
        description="Dictionary containing the image's geographic bounding box ('image_bbox': [TopLat,...]) and the GeoJSON of detected object centroids ('image_bbox_centers': FeatureCollection)",
        example={
            "image_bbox": [52.077, 5.182, 52.057, 5.223],
            "image_bbox_centers": {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"class": "car", "confidence": 0.9}, "geometry": {"type": "Point", "coordinates": [5.19, 52.06]}}
                 ] # Simplified example feature
             }
        }
    )

# Optional: Keep other models if needed elsewhere
class DetectionResult(BaseModel):
    bbox: List[float]
    class_name: str
    confidence: float