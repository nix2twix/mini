import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.measure import label

def process_tws(image: Image.Image) -> Image.Image:
    """
    Convert an RGB mask into a labeled grayscale mask using connected components.
    """
    image = image.convert("RGB")
    arr = np.array(image)

    green = np.array([79, 255, 130])
    red = np.array([255, 0, 0])

    # Binary mask
    binary = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    mask_red = np.all(arr == red.reshape(1,1,3), axis=-1)
    binary[mask_red] = 1

    # Label connected components
    labeled = label(binary, connectivity=2)

    # Sort labels top-bottom, left-right
    coords = []
    for i in range(1, labeled.max()+1):
        ys, xs = np.where(labeled == i)
        coords.append((i, ys.mean(), xs.mean()))
    coords.sort(key=lambda x: (x[1], x[2]))  # сортировка по среднему y, затем x

    # Перенумеруем labels
    new_labeled = np.zeros_like(labeled)
    for new_label, (old_label, _, _) in enumerate(coords, start=1):
        new_labeled[labeled == old_label] = new_label

    # Normalize to 0-255
    if new_labeled.max() > 0:
        labeled_img = (new_labeled / new_labeled.max() * 255).astype(np.uint8)
    else:
        labeled_img = new_labeled.astype(np.uint8)

    return Image.fromarray(labeled_img, mode="L")

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

def makeBlobs(labeledMask):
    mask_full = np.array(labeledMask)
    BLOBs = []             
    unique_labels = np.unique(mask_full)
    unique_labels = unique_labels[unique_labels > 0]

    for lbl in unique_labels:
        mask = (mask_full == lbl).astype(np.uint8) * 255  # OpenCV любит 0/255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                  

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])

                # Расстояния до центра
                points = contour.reshape(-1, 2)
                distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)

                radius_metrics = {
                    'average_radius': np.mean(distances),
                    'median_radius':  np.median(distances),
                    'min_radius':     np.min(distances),
                    'max_radius':     np.max(distances),
                    'std_radius':     np.std(distances),
                    'effective_radius': np.sqrt(cv2.contourArea(contour) / np.pi),
                    'bounding_circle_radius': cv2.minEnclosingCircle(contour)[1]
                }

                BLOBs.append([y, x, radius_metrics['average_radius']])

    temp_blobs = np.asarray(BLOBs)
    return temp_blobs
