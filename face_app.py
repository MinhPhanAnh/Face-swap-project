import sys
import cv2
from insightface.app import FaceAnalysis
from config import DETECTION_MODEL_NAME, DET_SIZE, PROVIDERS

def read_or_die(p: str):
    im = cv2.imread(p)
    if im is None:
        print(f"[ERROR] Can't read the image: {p}")
        sys.exit(1)
    return im

def pad(img, ratio=0.12):
    h, w = img.shape[:2]
    t = int(max(h, w) * ratio)
    return cv2.copyMakeBorder(img, t, t, t, t, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def best_face(faces):
    if not faces:
        return None
    best, best_val = None, -1.0
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        area = max(1.0, (x2 - x1) * (y2 - y1))
        score = float(getattr(f, "det_score", 0.9))
        val = area * score
        if val > best_val:
            best, best_val = f, val
    return best

def init_face_app():
    """Create FaceAnalysis instance."""
    app = FaceAnalysis(name=DETECTION_MODEL_NAME, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    try:
        app.models['detection'].thresh = 0.25  # Decrease detection threshold for better recall
    except Exception:
        pass
    return app

# Initialize FaceAnalysis app
app = init_face_app()

def quick_detect(img):
    """
    Try different variants of the image to detect faces.
    Return the first successful detection along with the variant tag.
    """
    variants = [
        ("angle", img),
        ("flip", cv2.flip(img, 1)),
        ("pad", pad(img, 0.12)),
        ("rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
    ]
    for tag, im in variants:
        faces = app.get(im)
        if faces:
            return im, faces, tag
    return None, [], "none"
