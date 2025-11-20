# main.py
"""
Main entry point for the face swapper application.
It initializes the face swap pipeline and processes the input images.
"""

from config import SRC_PATH, DST_PATH, OUT_PATH
from swapper_pipeline import FaceSwapPipeline

def main():
    pipeline = FaceSwapPipeline()
    pipeline.run(SRC_PATH, DST_PATH, OUT_PATH)

if __name__ == "__main__":
    main()
