# FaceSwap Pipeline

A modular face-swapping pipeline built with InsightFace.
This project swaps a source identity onto a target face while preserving pose, expression, and lighting from the target image.

# Features

* Face detection using InsightFace (buffalo_l model).
* Identity swap using INSwapper (inswapper_128.onnx).
* Automatic face selection with robust detection variants (original, flipped, padded, rotated).
* Post-processing for improved realism:
* Color transfer in Lab color space.
* Elliptical mask blending to reduce visible edges.
* Output normalization with resize + padding (default 640×640).
* Clean modular structure suitable for production projects.

# How the Pipeline Works
1. Face Detection

face_app.py uses the InsightFace analysis pipeline:

* Multiple detection variants are tested (original, flip, pad, rotation).

* The face with the best score-area heuristic is selected.

2. Face Swapping

The INSwapper model performs identity replacement:

* Target pose, lighting, and expression are preserved.

* Only the identity (embedding) is replaced.

This approach ensures historically accurate portrait style is retained when swapping onto paintings or statues.

3. Post-processing

To reduce the common artifacts seen in raw face-swapping:

* Color Transfer
  * Adjusts the swapped patch to match the target face’s overall tone.

* Elliptical Mask Blending
  * Softens boundaries using Gaussian-blurred masks, preventing “sticker-like” appearance.

4. Output Normalization

The final image is resized while preserving aspect ratio and padded to a square resolution (default 640×640).
This simplifies downstream usage in datasets, social media exports, or ML pipelines.
