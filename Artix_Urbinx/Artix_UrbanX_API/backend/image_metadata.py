
from PIL import Image
from PIL.ExifTags import TAGS
from pyproj import Transformer
import ast
import logging
import io  # Import io

logger = logging.getLogger(__name__)

def load_jpg_metadata(image_bytes: bytes): # Changed input to bytes
    """Loads metadata from a JPG image in memory."""
    try:
        # Use io.BytesIO to treat the image bytes as a file
        with Image.open(io.BytesIO(image_bytes)) as img:
            exif_data = img.getexif()
            metadata = {}

            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name in ["ImageDescription", "UserComment"]:
                        try:
                            metadata[tag_name] = ast.literal_eval(value) if isinstance(value, str) else value
                        except (SyntaxError, ValueError):
                            metadata[tag_name] = value
            return metadata
    except Exception as e:
        logger.exception(f"Error loading metadata from image bytes")
        return {}

class CoordinateTransformer:
    """
    A class to handle coordinate transformations and bounding box calculations.
    """

    def __init__(self, from_crs="EPSG:28992", to_crs="EPSG:4326"):
        self.transformer = Transformer.from_crs(from_crs, to_crs)

    def epsg_to_latlon(self, x, y, from_crs=None, to_crs=None):
        """
        Convert coordinates from the source CRS to latitude and longitude (EPSG:4326).

        :param x: X coordinate in the source CRS.
        :param y: Y coordinate in the source CRS.
        :param from_crs: Optional. The source CRS. Default is the CRS set in the constructor.
        :param to_crs: Optional. The target CRS. Default is the CRS set in the constructor.
        :return: (latitude, longitude) in the target CRS.
        """
        if from_crs and to_crs:
            transformer = Transformer.from_crs(from_crs, to_crs)
        else:
            transformer = self.transformer

        return transformer.transform(x, y)

    def calculate_bottom_right(self, top_left_x, top_left_y, width_px, height_px, pixel_size):
        """
        Calculate the bottom-right corner coordinates in the source CRS.

        :param top_left_x: X coordinate of the top-left corner in the source CRS.
        :param top_left_y: Y coordinate of the top-left corner in the source CRS.
        :param width_px: Image width in pixels.
        :param height_px: Image height in pixels.
        :param pixel_size: Pixel size in meters.
        :return: (bottom_right_x, bottom_right_y) coordinates in the source CRS.
        """
        bottom_right_x = top_left_x + (width_px * pixel_size)
        bottom_right_y = top_left_y - (height_px * pixel_size)
        return bottom_right_x, bottom_right_y

    def calculate_bbox_and_center(self, top_left_x, top_left_y, width_px, height_px, pixel_size, from_crs=None, to_crs=None):
        """
        Calculate the bounding box with paired corners and the center in latitude and longitude.

        :param top_left_x: X coordinate of the top-left corner in the source CRS.
        :param top_left_y: Y coordinate of the top-left corner in the source CRS.
        :param width_px: Image width in pixels.
        :param height_px: Image height in pixels.
        :param pixel_size: Pixel size in meters.
        :param from_crs: Optional. The source CRS. Default is the CRS set in the constructor.
        :param to_crs: Optional. The target CRS. Default is the CRS set in the constructor.
        :return: Dictionary with 'center' and 'bounding_box':
                 {
                     "center": (center_lat, center_lon),
                     "bounding_box": [min_lon, min_lat, max_lon, max_lat],
                     "corner_points": [top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon]
                 }
        """
        # Calculate other corners in source CRS
        bottom_right_x, bottom_right_y = self.calculate_bottom_right(
            top_left_x, top_left_y, width_px, height_px, pixel_size
        )
        top_right_x = bottom_right_x
        top_right_y = top_left_y
        bottom_left_x = top_left_x
        bottom_left_y = bottom_right_y

        # Convert all corners to lat/lon
        top_left_lat, top_left_lon = self.epsg_to_latlon(top_left_x, top_left_y, from_crs, to_crs)
        top_right_lat, top_right_lon = self.epsg_to_latlon(top_right_x, top_right_y, from_crs, to_crs)
        bottom_left_lat, bottom_left_lon = self.epsg_to_latlon(bottom_left_x, bottom_left_y, from_crs, to_crs)
        bottom_right_lat, bottom_right_lon = self.epsg_to_latlon(bottom_right_x, bottom_right_y, from_crs, to_crs)

        # Calculate center of the bounding box in source CRS
        center_x = (top_left_x + bottom_right_x) / 2
        center_y = (top_left_y + bottom_right_y) / 2
        center_lat, center_lon = self.epsg_to_latlon(center_x, center_y, from_crs, to_crs)

        # Return dictionary with center and bounding box
        return {
            "center": [center_lon,center_lat],
            "bounding_box": [top_left_lat,top_left_lon, bottom_right_lat,bottom_right_lon],
            "corner_points_poly": [(top_left_lon, top_left_lat),(top_right_lon, top_right_lat),(bottom_right_lon, bottom_right_lat),(bottom_left_lon, bottom_left_lat),(top_left_lon, top_left_lat) ]}