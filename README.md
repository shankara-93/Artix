
## Prerequisites

*   **Python 3.9+:** Ensure you have Python 3.9 or a later version installed.
*   **Docker:** Install Docker Desktop or Docker Engine on your system.

## Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd my_project
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate.bat  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Your YOLO Model:**

    *   Download your pre-trained YOLO model (e.g., `best.pt`) and place it in the `model/` directory.  Make sure to replace `best.pt` with the actual name of your model file.

5.  **Configure Model Path:**

    *   Update the `model_path` in `config.py` to point to the correct location of your model file. By default, it is configured as:
        ```python
        self.model_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "best.pt")
        ```
        This path is relative to the project root.  You can also set an environment variable named `MODEL_PATH` and it will take that value.

## API Configuration

*   **Image Metadata:**  The API expects input images in JPG format with specific EXIF metadata. The following metadata keys should be present in either the `ImageDescription` or `UserComment` EXIF tag:

    *   `top_left_x`: X coordinate of the top-left corner of the image (in the source CRS).
    *   `top_left_y`: Y coordinate of the top-left corner of the image (in the source CRS).
    *   `width_px`: Image width in pixels.
    *   `height_px`: Image height in pixels.
    *   `pixel_size`: Pixel size (in meters or other appropriate units for your CRS).
    *   `crs`: The coordinate reference system (e.g., "EPSG:28992").

    **Example Metadata:**

    ```json
    {
        "top_left_x": 155000.0,
        "top_left_y": 463000.0,
        "width_px": 800,
        "height_px": 600,
        "pixel_size": 0.15,
        "crs": "EPSG:28992"
    }
    ```

## Running the Application

1.  **Start the FastAPI Application (Without Docker):**

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

    This will start the API server on `http://localhost:8000`.

## Docker Instructions

1.  **Build the Docker Image:**

    ```bash
    docker build -t my-detection-app .
    ```

    Make sure you are in the root directory of your project (`my_project/`) when you run this command.

2.  **Run the Docker Container:**

    ```bash
    docker run -p 8000:8000 my-detection-app
    ```

    This will run the container and map port 8000 on your host machine to port 8000 inside the container. You can then access your API at `http://localhost:8000`.

## API Endpoint

*   **`/detect/` (POST):**

    *   **Input:**  A JPG image file sent as `multipart/form-data`.  The image must contain the EXIF metadata as described above.
    *   **Output:** A GeoJSON FeatureCollection representing the detected objects as points with class names and confidence scores.

    **Example `curl` Request:**

    ```bash
    curl -X POST \
      "http://localhost:8000/detect/" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/your/image.jpg"
    ```

    **(Replace `/path/to/your/image.jpg` with the actual path to your image file.)**

    **Example GeoJSON Response:**

    ```json
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "geometry": {
            "type": "Point",
            "coordinates": [
              -122.4194,
              37.7749
            ]
          },
          "properties": {
            "class": "car",
            "confidence": 0.95
          }
        },
        ...
      ]
    }
    ```

## Testing

(Optional) Add instructions for running unit and integration tests here if you have them.

## Contributing

(Optional) Add instructions for contributing to the project.

## License

(Optional) Add license information.