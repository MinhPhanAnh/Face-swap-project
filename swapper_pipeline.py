# swapper_pipeline.py
import cv2
from insightface.model_zoo import get_model

from config import INSWAPPER_MODEL_PATH, INSWAPPER_DOWNLOAD, PROVIDERS, FINAL_SIZE
from face_app import app, quick_detect, best_face, read_or_die
from color_utils import apply_face_color_match
from image_utils import resize_with_padding

class FaceSwapPipeline:
    def __init__(self):
        self.swapper = get_model(
            INSWAPPER_MODEL_PATH,
            download=INSWAPPER_DOWNLOAD,
            providers=PROVIDERS
        )

    def load_images(self, src_path, dst_path):
        img_src = read_or_die(src_path)
        img_dst = read_or_die(dst_path)
        return img_src, img_dst

    def detect_source_face(self, img_src):
        src_img, src_faces, src_tag = quick_detect(img_src)
        if not src_faces:
            raise RuntimeError(
                "[ERROR] No face detected in source image (try brighter/more frontal image)."
            )
        src_face = best_face(src_faces)
        print(f"[SRC] {src_tag} -> {len(src_faces)} face. Use the best one.")
        return src_img, src_face

    def detect_target_faces(self, img_dst):
        dst_img, dst_faces, dst_tag = quick_detect(img_dst)
        if not dst_faces:
            raise RuntimeError("[ERROR] No face detected in target image.")
        print(f"[DST] {dst_tag} -> {len(dst_faces)} mặt.")
        return dst_img, dst_faces

    def swap_faces(self, src_face, dst_img, dst_faces):
        """
        Swap the src_face onto all faces in dst_img.
        Return the resulting image.
        """
        result = dst_img.copy()
        for i, face in enumerate(dst_faces, 1): 
            result = self.swapper.get(result, face, src_face, paste_back=True)
            result = apply_face_color_match(result, dst_img, face)
            print(f"[OK] swap #{i}")
        return result

    def postprocess_output(self, img):
        return resize_with_padding(img, size=FINAL_SIZE)

    def run(self, src_path, dst_path, out_path):
        img_src, img_dst = self.load_images(src_path, dst_path)

        src_img, src_face = self.detect_source_face(img_src)
        dst_img, dst_faces = self.detect_target_faces(img_dst)

        result = self.swap_faces(src_face, dst_img, dst_faces)
        result_std = self.postprocess_output(result)

        cv2.imwrite(out_path, result_std)
        print(f"✅ Saved {out_path} with the size {FINAL_SIZE}x{FINAL_SIZE}")
