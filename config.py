import os
from ultralytics import YOLO
import supervision as sv

class Settings:
    def __init__(self):
        # Construct the path to the model file relative to the project root
        # Get model path from environment variable, or use default if not set
        # --- FIX IS HERE: Removed one os.path.dirname ---
        project_root = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(project_root, "model", "best.pt")

        self.model_path: str = os.environ.get("MODEL_PATH", default_model_path)
        self.model = self.load_model()
        self.confidence_threshold: float = 0.5

    def load_model(self):
        """Load model from weights"""
        # --- ADDED CHECK: Verify the path exists before trying to load ---
        if not os.path.exists(self.model_path):
             print(f"Error: Model file not found at the calculated path: {self.model_path}")
             return None
        try:
            print(f"Loading model from: {self.model_path}")
            return YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading the model: {e}")
            return None

settings = Settings()