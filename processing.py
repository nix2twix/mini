import numpy as np
from PIL import Image, ImageDraw

def process_tws(image: Image.Image) -> Image.Image:
    """
    Convert an RGB mask into a binary mask.
    """
    image = image.convert("RGB")
    arr = np.array(image)
    
    green = np.array([79, 255, 130])
    red = np.array([255, 0, 0])
    out = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)

    mask_green = np.all(arr == green, axis=-1)
    out[mask_green] = 0

    mask_red = np.all(arr == red, axis=-1)
    out[mask_red] = 255

    return Image.fromarray(out, mode="L")

def process_cellpose(image: Image.Image) -> Image.Image:
    """
    Convert a label mask from Cellpose (.tif) to a visible grayscale mask.
    """
    arr = np.array(image)
    arr = arr.astype("int32")
    max_val = arr.max() if arr.max() > 0 else 1
    norm = (arr / max_val) * 255
    norm = norm.astype("uint8")

    return Image.fromarray(norm, mode="L")

def process_dlgram(json_data: dict) -> Image.Image:
    """
    Convert DLgram JSON to a labeled grayscale mask. 
    """

    width = json_data.get("imageWidth")
    height = json_data.get("imageHeight")
    # Create blank mask
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)

    label_value = 1

    for shape in json_data.get("shapes", []):
        if shape.get("label") != "nanoparticle":
            continue
        points = [(int(x), int(y)) for x, y in shape.get("points", [])]
        drawer.polygon(points, fill=label_value)
        label_value += 1

    # Normalize label values to 0–255
    arr = np.array(mask)
    max_val = arr.max() if arr.max() > 0 else 1
    arr = (arr / max_val * 255).astype("uint8")
    
    return Image.fromarray(arr, mode="L")