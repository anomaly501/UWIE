# file: detector.py [SIMPLIFIED & FINAL VERSION]

from pathlib import Path
import cv2
from ultralytics import YOLO

def draw_boxes(frame, boxes, scores, classes, names):
    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]); color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{names.get(int(cls), str(cls))}'
        t_size = cv2.getTextSize(label, 0, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def get_detections(results):
    detections_found = False; boxes, scores, classes = [], [], []
    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        detections_found = True
        for box in results[0].boxes:
            boxes.append(box.xyxy.tolist()[0])
            scores.append(float(box.conf.tolist()[0]))
            classes.append(int(box.cls.tolist()[0]))
    return detections_found, boxes, scores, classes

def run_inference(video_path: str, out_path: str):
    model_path = str(Path(__file__).resolve().parent / 'models' / 'last.pt')
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    names = {i: n for i, n in enumerate(model.names)} if hasattr(model, 'names') else {}
    any_detection_in_video = False

    while True:
        ret, frame = cap.read();
        if not ret: break
        
        results = model.predict(frame, conf=0.45, imgsz=640, verbose=False)
        detections_found, boxes, scores, classes = get_detections(results)
        
        if detections_found:
            any_detection_in_video = True
            draw_boxes(frame, boxes, scores, classes, names)
        out.write(frame)

    cap.release(); out.release()
    # अब यह सिर्फ True या False लौटाएगा
    return any_detection_in_video