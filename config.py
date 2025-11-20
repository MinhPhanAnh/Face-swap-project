SRC_PATH = "path_to_your_source_image"         
DST_PATH = "path_to_your_character_image"   
OUT_PATH = "output_image"               

# Output image size
FINAL_SIZE = 640

# Model for face detection
DETECTION_MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
PROVIDERS = ['CPUExecutionProvider']

# Model faceswap
INSWAPPER_MODEL_PATH = "path_to_your_faceswap_model"
INSWAPPER_DOWNLOAD = True
