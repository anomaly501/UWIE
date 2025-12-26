import os
import time
import argparse
import cv2
import torch
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
# import metrics if available
try:
    from nets.metrics import psnr as metric_psnr, ssim as metric_ssim, uiqm as metric_uiqm
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False


parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default=r".\test.mp4")
parser.add_argument("--output_path", type=str, default="output_video.mp4")
parser.add_argument("--model_name", type=str, default="amangan")  
parser.add_argument("--model_path", type=str, default=r".\models\BEST.pth")
opt = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if opt.model_name.lower() == 'amangan':
    from nets import amangan
    model = amangan.GeneratorAmanGAN()
else:
    raise ValueError("Unknown model name")

model.load_state_dict(torch.load(opt.model_path, map_location=device))
model.to(device)
model.eval()
print(f"Loaded {opt.model_name} model from {opt.model_path}")


img_width, img_height = 256, 256
transform = transforms.Compose([
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


cap = cv2.VideoCapture(opt.video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video {opt.video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(opt.output_path, fourcc, fps, (width, height))

times = []
# metrics accumulators
psnr_vals = []
ssim_vals = []
uiqm_vals = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp_tensor = transform(frame_pil).unsqueeze(0).to(device)

    # Enhance frame
    start = time.time()
    with torch.no_grad():
        gen_tensor = model(inp_tensor)
    times.append(time.time() - start)

    gen_img = gen_tensor.squeeze(0).cpu()
    gen_img = (gen_img * 0.5 + 0.5).clamp(0, 1)  
    gen_img = transforms.ToPILImage()(gen_img)
    gen_img = cv2.cvtColor(np.array(gen_img), cv2.COLOR_RGB2BGR)

    gen_img = cv2.resize(gen_img, (width, height))
    out.write(gen_img)

    # compute metrics vs original frame if metrics are present
    if _HAS_METRICS:
        try:
            # ensure same dtype/shape
            orig = frame
            enhanced = gen_img
            ps = metric_psnr(orig, enhanced)
            ss = metric_ssim(orig, enhanced)
            ui = metric_uiqm(enhanced)
            psnr_vals.append(ps)
            ssim_vals.append(ss)
            uiqm_vals.append(ui)
        except Exception:
            # non-fatal: skip metric for this frame
            pass

cap.release()
out.release()


if len(times) > 1:
    Ttime = sum(times[1:])
    Mtime = sum(times[1:]) / len(times[1:])
    print(f"Total frames: {len(times)}")
    print(f"Time taken: {Ttime:.2f} sec at {1./Mtime:.2f} fps")
print(f"Saved enhanced video to {opt.output_path}")

if _HAS_METRICS and len(psnr_vals) > 0:
    import statistics
    print('--- Metrics summary ---')
    print(f'Frames measured: {len(psnr_vals)}')
    print(f'PSNR  mean: {statistics.mean(psnr_vals):.3f}, std: {statistics.pstdev(psnr_vals):.3f}')
    print(f'SSIM  mean: {statistics.mean(ssim_vals):.4f}, std: {statistics.pstdev(ssim_vals):.4f}')
    print(f'UIQM  mean: {statistics.mean(uiqm_vals):.3f}, std: {statistics.pstdev(uiqm_vals):.3f}')

# --- Automatically run detector on the enhanced video so the final output is annotated
try:
    from pathlib import Path
    # import the detector function from detect_video.py (must be in same project)
    from detect_video import run_inference

    enhanced_path = Path(opt.output_path)
    # default detector model inside this project (same logic as detect_video.main)
    detector_model = str(Path(__file__).resolve().parent / 'models' / 'last.pt')
    # write detector output next to enhanced file with suffix
    detector_out = enhanced_path.with_name(enhanced_path.stem + '_detected' + enhanced_path.suffix)

    print(f"Running detector on enhanced video: {enhanced_path}")
    # pass the original input video as reference so PSNR/SSIM can be measured
    orig_video_path = Path(opt.video_path)
    run_inference(enhanced_path, detector_model, detector_out, ref_path=orig_video_path)
    print(f"Saved annotated (detected) video to {detector_out}")
except Exception as _e:
    # Keep behaviour non-fatal if detector fails (e.g., ultralytics not installed or model missing)
    print(f"Detector step skipped or failed: {_e}")
