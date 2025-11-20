import cv2
import numpy as np

def color_transfer(src_patch, dst_patch):
    """
    Adjust the color of src_patch to match that of dst_patch using
    the mean and standard deviation in the LAB color space.
    """
    if src_patch.size == 0 or dst_patch.size == 0:
        return src_patch

    src = cv2.cvtColor(src_patch, cv2.COLOR_BGR2LAB).astype("float32")
    dst = cv2.cvtColor(dst_patch, cv2.COLOR_BGR2LAB).astype("float32")

    src_mean, src_std = cv2.meanStdDev(src)
    dst_mean, dst_std = cv2.meanStdDev(dst)

    src_mean = src_mean.reshape((1, 1, 3))
    src_std  = src_std.reshape((1, 1, 3))
    dst_mean = dst_mean.reshape((1, 1, 3))
    dst_std  = dst_std.reshape((1, 1, 3))

    src_std = np.maximum(src_std, 1e-6)

    out = (src - src_mean) / src_std * dst_std + dst_mean
    out = np.clip(out, 0, 255).astype("uint8")
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out

def apply_face_color_match(result_img, dst_img, face, margin_ratio=0.25):
    """
    Apply color correction to the face region in result_img
    to match the color of the corresponding region in dst_img.
    
    Parameters:
    - result_img: The image with the swapped face.
    - dst_img: The target image to match colors from.
    - face: The face object containing bounding box information.
    - margin_ratio: The ratio to expand the bounding box for color correction.
    """
    h, w = result_img.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)

    fw = x2 - x1
    fh = y2 - y1
    cx = x1 + fw // 2
    cy = y1 + fh // 2
    r = int(max(fw, fh) * (1.0 + margin_ratio) / 2.0)

    x1n = max(0, cx - r)
    y1n = max(0, cy - r)
    x2n = min(w, cx + r)
    y2n = min(h, cy + r)

    if x2n <= x1n or y2n <= y1n:
        return result_img

    src_patch = result_img[y1n:y2n, x1n:x2n]
    dst_patch = dst_img[y1n:y2n, x1n:x2n]

    if src_patch.shape != dst_patch.shape:
        return result_img

    corrected = color_transfer(src_patch, dst_patch)

    ph, pw = src_patch.shape[:2]
    mask = np.zeros((ph, pw), dtype="float32")
    center = (pw // 2, ph // 2)
    axes = (int(pw * 0.45), int(ph * 0.55))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    mask_3c = np.dstack([mask] * 3)

    blended = mask_3c * corrected + (1.0 - mask_3c) * src_patch
    blended = blended.astype("uint8")

    result_img[y1n:y2n, x1n:x2n] = blended
    return result_img
