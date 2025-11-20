# image_utils.py
import cv2

def resize_with_padding(img, size=640):
    """
    Resize image to fit in a square of given size with padding.
    The image is resized to maintain aspect ratio, then padded with black borders.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    scale = float(size) / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded
