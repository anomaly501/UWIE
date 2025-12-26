# file: video_processor.py [CORRECTED & FINAL VERSION]

import cv2
import torch
import statistics
from torchvision import transforms
from PIL import Image
from pathlib import Path
# --- यहाँ सुधार है: NumPy को इम्पोर्ट करें ---
import numpy as np

from detector import run_inference as run_yolo_detector

try:
    from nets.metrics import psnr as metric_psnr, ssim as metric_ssim, uiqm as metric_uiqm
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

def process_and_detect(video_path_str: str, output_dir_str: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path(__file__).resolve().parent
    
    enhancer_model_path = str(base_path / 'models' / 'BEST.pth')
    from nets import amangan
    model = amangan.GeneratorAmanGAN()
    model.load_state_dict(torch.load(enhancer_model_path, map_location=device))
    model.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    
    enhanced_path = str(output_dir / f"{video_path.stem}_enhanced.mp4")
    final_annotated_path = str(output_dir / f"{video_path.stem}_final.mp4")
    
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path_str}")
        
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(enhanced_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    
    input_uiqm_vals, enhanced_psnr_vals, enhanced_ssim_vals, output_uiqm_vals = [], [], [], []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if _HAS_METRICS:
            try:
                input_uiqm_vals.append(metric_uiqm(frame))
            except Exception: pass
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp_tensor = transform(frame_pil).unsqueeze(0).to(device)
        
        with torch.no_grad(): gen_tensor = model(inp_tensor)
        
        gen_img = gen_tensor.squeeze(0).cpu(); gen_img = (gen_img * 0.5 + 0.5).clamp(0, 1)
        gen_img_pil = transforms.ToPILImage()(gen_img)
        # np.array() का उपयोग यहाँ होता है, इसलिए ऊपर इम्पोर्ट करना ज़रूरी है
        gen_img_cv2 = cv2.cvtColor(np.array(gen_img_pil), cv2.COLOR_RGB2BGR)
        gen_img_resized = cv2.resize(gen_img_cv2, (w, h))
        out.write(gen_img_resized)
        
        if _HAS_METRICS:
            try:
                enhanced_psnr_vals.append(metric_psnr(frame, gen_img_resized))
                enhanced_ssim_vals.append(metric_ssim(frame, gen_img_resized))
                output_uiqm_vals.append(metric_uiqm(gen_img_resized))
            except Exception: pass
            
    cap.release(); out.release()
    
    detection_found = run_yolo_detector(enhanced_path, final_annotated_path)
    
    final_metrics = {}
    try:
        if _HAS_METRICS:
            final_metrics["psnr"] = {"input": "N/A", "output": f"{statistics.mean(enhanced_psnr_vals):.2f} ± {statistics.pstdev(enhanced_psnr_vals):.2f}"}
            final_metrics["ssim"] = {"input": "N/A", "output": f"{statistics.mean(enhanced_ssim_vals):.3f} ± {statistics.pstdev(enhanced_ssim_vals):.3f}"}
            final_metrics["uiqm"] = {
                "input": f"{statistics.mean(input_uiqm_vals):.2f} ± {statistics.pstdev(input_uiqm_vals):.2f}",
                "output": f"{statistics.mean(output_uiqm_vals):.2f} ± {statistics.pstdev(output_uiqm_vals):.2f}"
            }
    except Exception as e:
        print(f"Could not compute all metrics: {e}")

    return Path(final_annotated_path).name, final_metrics, detection_found