# Underwater Image Enhancement (MVS)

Flask-based web app that enhances underwater video frames with AmanGAN, computes quality metrics (PSNR/SSIM/UIQM), and runs YOLOv8 detection to overlay targets. Upload a video or image, review a browser-safe preview of the input, and download the annotated result.

## Features
- Underwater enhancement via `nets/amangan.GeneratorAmanGAN` using `models/BEST.pth`.
- YOLOv8 inference (`models/last.pt`) with bounding-box overlays on the enhanced output.
- Quality metrics (PSNR, SSIM, UIQM) when metric utilities are available in `nets/metrics.py`.
- Browser-safe H.264 MP4 copy of uploads for reliable playback in the UI.
- Web UI with side-by-side input/output playback and status alerts.

## Project Layout
- `app.py`: Flask server, upload handling, conversion to browser-safe MP4, and orchestration.
- `video_processor.py`: Frame-by-frame enhancement, metric aggregation, and detection pass.
- `detector.py`: YOLOv8 inference and box drawing on processed frames.
- `templates/index.html` + `static/styles.css`: Mission-themed UI.
- `models/`: Expected to contain `BEST.pth` (enhancer) and `last.pt` (detector).
- `uploads/`, `outputs/`: Auto-created staging/output folders.

## Prerequisites
- Python 3.10+ recommended.
- GPU with CUDA for best throughput (CPU works but will be slow).
- FFmpeg codecs are bundled through OpenCV; H.264 (`avc1`) writing relies on your OpenCV build.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place model weights in `models/`:
   - `BEST.pth` (AmanGAN generator)
   - `last.pt` (YOLOv8 detector)

## Run the Web App
```bash
python app.py
```
- Server defaults to `http://127.0.0.1:8000`.
- Supported uploads: `mp4, mov, avi, mkv, webm, jpg, jpeg, png`.
- The app saves the original file to `uploads/`, produces a browser-safe copy for preview, writes the enhanced video to `outputs/<name>_enhanced.mp4`, and the annotated final video to `outputs/<name>_final.mp4`.

## CLI Test Script
For offline processing without the UI, run:
```bash
python testvideo.py --video_path <input> --output_path <output_video.mp4>
```
The script enhances frames, optionally computes metrics (if available), and attempts to run detection on the enhanced output.

## Notes
- Metrics are optional; if `nets/metrics.py` is missing or errors, processing continues without them.
- Default detection confidence is 0.45 and image size 640 (see `detector.py`).
- If playback fails in the browser, ensure the H.264 codec is available in your OpenCV build.

## Troubleshooting
- `Cannot open video`: confirm the path and codec support; try re-encoding the input.
- `Failed to process the uploaded video file`: upload will abort if the browser-safe conversion step fails; check console logs for OpenCV codec errors.
- Slow performance: prefer running on GPU; lower resolution or frame rate to reduce load.
