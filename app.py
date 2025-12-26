# file: app.py [ULTIMATE FOOLPROOF & FINAL VERSION]

import os
import cv2 # OpenCV को इम्पोर्ट करें
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from video_processor import process_and_detect

# --- ऐप का सेटअप ---
app = Flask(__name__)

# --- फोल्डर कॉन्फ़िगरेशन ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'jpg', 'jpeg', 'png'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- नया हेल्पर फंक्शन: वीडियो को ब्राउज़र-फ्रेंडली MP4 में बदलें ---
def convert_to_browser_safe_mp4(input_path, output_path):
    """
    किसी भी वीडियो को पढ़ता है और उसे ब्राउज़र-फ्रेंडली H.264 (avc1) MP4 में सेव करता है.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video for conversion: {input_path}")
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # ब्राउज़र के लिए सबसे अच्छा कोडेक
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
        cap.release()
        out.release()
        print(f"Successfully converted {input_path} to browser-safe {output_path}")
        return True
    except Exception as e:
        print(f"Error during video conversion: {e}")
        return False

# --- मुख्य पेज का रूट ---
@app.route('/')
def index():
    return render_template('index.html')

# --- फाइल अपलोड और प्रोसेसिंग का रूट ---
@app.route('/upload', methods=['POST'])
def upload_and_process():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    original_input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(original_input_path)
    
    # --- यहाँ मुख्य सुधार है ---
    # ब्राउज़र में दिखाने के लिए एक सुरक्षित कॉपी बनाएँ
    safe_filename = f"safe_{os.path.splitext(filename)[0]}.mp4"
    browser_safe_input_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    
    # वीडियो को कन्वर्ट करें
    conversion_success = convert_to_browser_safe_mp4(original_input_path, browser_safe_input_path)
    
    if not conversion_success:
        return jsonify({'error': 'Failed to process the uploaded video file.'}), 500

    try:
        # मुख्य प्रोसेसिंग के लिए हमेशा ओरिजिनल, हाई-क्वालिटी फाइल का उपयोग करें
        final_video_name, metrics, detected = process_and_detect(original_input_path, app.config['OUTPUT_FOLDER'])
        
        return jsonify({
            'success': True,
            # ब्राउज़र को हमेशा "सुरक्षित" वाली फाइल का URL भेजें
            'original_url': f'/uploads/{safe_filename}',
            'output_url': f'/outputs/{final_video_name}',
            'metrics': metrics,
            'detection_found': detected
        })
    except Exception as e:
        print(f"FATAL ERROR during processing: {e}")
        return jsonify({'error': 'An error occurred during video processing.'}), 500

# --- आउटपुट वीडियो दिखाने के लिए रूट ---
@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# --- इनपुट वीडियो दिखाने के लिए रूट ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- सर्वर को शुरू करें ---
if __name__ == '__main__':
    app.run(debug=True, port=8000)