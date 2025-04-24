
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from PIL import Image # Keep PIL import if needed elsewhere, but not strictly for numpy conversion
from typing import Dict, Any, Tuple, Optional # Use Optional for clarity
import logging

logger = logging.getLogger(__name__) # Use module-specific logger

def scaleCCC(x: np.ndarray) -> np.ndarray:
    """Percentile scaling (2nd to 98th percentile) for a NumPy array."""
    # Ensure input is float for percentile calculations
    x = x.astype(np.float32)
    # Handle potential all-NaN slices gracefully
    lower = np.nanpercentile(x, 2)
    upper = np.nanpercentile(x, 98)

    # Avoid division by zero or invalid scale range
    if upper <= lower:
        logger.warning(f"Scale calculation encountered upper percentile <= lower ({upper} <= {lower}). Returning zeros.")
        # Decide behavior: return zeros, ones, or original scaled by a fixed factor?
        # Returning zeros might be safest if scale is meaningless.
        return np.zeros_like(x, dtype=np.float32)

    scaled = (x - lower) / (upper - lower)
    # Clip to ensure values are within [0, 1] after scaling
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled

def load_raster_metadata_from_dataset(src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Loads key metadata directly from an open rasterio dataset."""
    try:
        # Basic check if dataset seems valid
        if not src or src.closed:
             logger.error("Attempted to load metadata from an invalid or closed rasterio dataset.")
             return {}

        transform = src.transform
        top_left_x, top_left_y = transform * (0, 0)  # Top-left corner coordinates
        width_px = src.width
        height_px = src.height
        pixel_size_x, pixel_size_y = src.res  # Pixel sizes (may be negative)
        crs = src.crs
        bounds = src.bounds

        metadata = {
            'width_px': width_px,
            'height_px': height_px,
            'crs': crs.to_string() if crs else None,
            'bounds': { # Using standard GeoJSON-like bbox order [min_x, min_y, max_x, max_y]
                'left': bounds.left,
                'bottom': bounds.bottom,
                'right': bounds.right,
                'top': bounds.top
            },
            'transform': list(transform), # Store affine transform components [a, b, c, d, e, f]
            'pixel_size_x': abs(pixel_size_x), # Often prefer positive representation
            'pixel_size_y': abs(pixel_size_y),
            'top_left_x': top_left_x, # Explicit top-left for potential use
            'top_left_y': top_left_y,
            'band_count': src.count,
            'dtypes': [str(dt) for dt in src.dtypes] # Data types of bands
        }
        logger.debug(f"Successfully extracted metadata: {metadata}")
        return metadata

    except AttributeError as ae:
        logger.error(f"Attribute error extracting metadata (likely invalid dataset object): {ae}")
        return {}
    except Exception as e:
        logger.exception(f"Unexpected error extracting metadata from raster dataset: {e}")
        return {} # Return empty dict on any error

def convert_tif_bytes_to_rgb_numpy(tif_bytes: bytes) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Converts TIF image bytes to an 8-bit RGB NumPy array (H, W, C) using rasterio.
    Also extracts metadata. Reads first 3 bands as RGB.

    Returns:
        Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
            (rgb_image_array, metadata) on success, (None, None) on failure.
    """
    rgb_image_array = None
    metadata = None
    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as src:
                # --- Extract Metadata ---
                metadata = load_raster_metadata_from_dataset(src)
                if not metadata:
                    # Proceeding without metadata might be okay for *conversion*, but not for georeferencing later.
                    logger.warning("Could not extract metadata from TIF. Conversion might proceed but georeferencing will fail.")
                    # Depending on strictness, you might return None, None here.
                    # metadata = {} # Or assign empty dict if proceeding is acceptable

                # --- Read and Scale Bands (Assuming 3 Bands for RGB) ---
                if src.count < 3:
                    logger.error(f"TIF file has only {src.count} bands, expected at least 3 for RGB conversion.")
                    return None, None # Cannot form RGB

                logger.info(f"Reading bands 1, 2, 3 from TIF with data types {src.dtypes[:3]} and shape {src.shape}")

                # Read bands (adjust indices if needed, e.g., some TIFs are BGR 1=B, 2=G, 3=R)
                # Applying scaleCCC band-wise
                # Ensure read data type doesn't cause issues with scaleCCC (which expects float)
                red_scaled = scaleCCC(src.read(1).astype(np.float32))   # Read band 1 as Red
                green_scaled = scaleCCC(src.read(2).astype(np.float32)) # Read band 2 as Green
                blue_scaled = scaleCCC(src.read(3).astype(np.float32))  # Read band 3 as Blue

                # --- Stack and Scale to uint8 ---
                # Stack bands into (Height, Width, Channels) format
                # Note: np.dstack creates (H, W, D), which is standard for image processing libs like OpenCV/PIL
                rgb_scaled = np.dstack((red_scaled, green_scaled, blue_scaled))

                # Scale [0, 1] float to [0, 255] uint8
                rgb_image_array = (rgb_scaled * 255).astype(np.uint8)

                logger.info(f"Successfully converted TIF bytes to RGB NumPy array with shape {rgb_image_array.shape}")
                return rgb_image_array, metadata

    except rasterio.RasterioIOError as e:
        logger.error(f"Rasterio Error processing TIF bytes: {e}. Is it a valid TIF file?")
        return None, None
    except MemoryError:
        logger.error("MemoryError during TIF processing. Image might be too large.")
        return None, None
    except Exception as e:
        logger.exception(f"Unexpected error during TIF conversion: {e}") # Log full traceback
        return None, None