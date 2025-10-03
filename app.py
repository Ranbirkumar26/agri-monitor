from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34
from ultralytics import YOLO
from PIL import Image
import threading
import time
import csv
import pickle
import random
from datetime import datetime
import io
import base64

app = Flask(__name__)

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------
leaf_yolo = YOLO("leaf_disease.pt")
weed_yolo = YOLO("weed.pt")

num_classes = 4
resnet_model = resnet34(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load("leaf_resnet.pth", map_location="cpu"))
resnet_model.eval()

with open("health_bundle.pkl", "rb") as f:
    soil_model = pickle.load(f)

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ["Blast", "Blight", "Brown_Spot", "Healthy"]

# -------------------------------------------------------
# GPS Mock Generator
# -------------------------------------------------------
class GPSGenerator:
    def __init__(self, base_lat=13.0827, base_lon=80.2707):
        self.base_lat = base_lat
        self.base_lon = base_lon
    
    def get_next_location(self):
        distance = random.uniform(0.00009, 0.00018)
        lat_offset = distance * random.choice([-1, 1]) * abs(random.uniform(0.5, 1.0))
        lon_offset = distance * random.choice([-1, 1]) * abs(random.uniform(0.5, 1.0))
        return (
            round(self.base_lat + lat_offset, 6),
            round(self.base_lon + lon_offset, 6)
        )

gps_gen = GPSGenerator()

# -------------------------------------------------------
# Result Logger
# -------------------------------------------------------
class ResultLogger:
    def __init__(self, filename="result.csv"):
        self.filename = filename
        self.lock = threading.Lock()
        self.recent_detections = []
        self.initialize_csv()
    
    def initialize_csv(self):
        try:
            with open(self.filename, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Type", "Result", "Latitude", "Longitude"])
        except FileExistsError:
            pass
    
    def log_result(self, result_type, result_value, gps_coords):
        with self.lock:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, result_type, result_value, gps_coords[0], gps_coords[1]])
            
            # Keep last 10 detections for display
            detection = {
                "timestamp": timestamp,
                "type": result_type,
                "result": result_value,
                "lat": gps_coords[0],
                "lon": gps_coords[1]
            }
            self.recent_detections.insert(0, detection)
            if len(self.recent_detections) > 10:
                self.recent_detections.pop()
    
    def get_recent_detections(self):
        with self.lock:
            return self.recent_detections.copy()

logger = ResultLogger()

# -------------------------------------------------------
# Soil Health Monitor
# -------------------------------------------------------
class SoilHealthMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
    
    def run(self):
        while self.running:
            time.sleep(60)
            try:
                with open("rand.csv", 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:
                        data_row = random.choice(rows[1:])
                        features = [float(x) for x in data_row]
                        prediction = soil_model.predict([features])[0]
                        gps = gps_gen.get_next_location()
                        logger.log_result("Soil_Health", f"Quality_{prediction}", gps)
                        print(f"[Soil] Quality: {prediction} at {gps}")
            except Exception as e:
                print(f"Soil monitoring error: {e}")

# -------------------------------------------------------
# Camera Processing
# -------------------------------------------------------
class CameraProcessor:
    def __init__(self):
        self.cap = None
        self.running = False
        self.phone_ip = None
        self.frame_count = 0
        self.process_every_n_frames = 10
        self.last_processed_frame = None
        self.lock = threading.Lock()
    
    def set_phone_ip(self, ip):
        self.phone_ip = ip
    
    def start_camera(self):
        if not self.phone_ip:
            return False, "Phone IP not set"
        
        camera_url = f"http://{self.phone_ip}/video"
        self.cap = cv2.VideoCapture(camera_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        time.sleep(1)
        
        if not self.cap.isOpened():
            return False, "Cannot connect to camera"
        
        self.running = True
        return True, "Camera started"
    
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        if not self.running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.resize(frame, (640, 480))
        self.frame_count += 1
        
        # Process AI models every N frames
        if self.frame_count % self.process_every_n_frames == 0:
            annotated_frame = self.process_detections(frame)
            with self.lock:
                self.last_processed_frame = annotated_frame
        else:
            with self.lock:
                if self.last_processed_frame is not None:
                    annotated_frame = self.last_processed_frame
                else:
                    annotated_frame = frame
        
        return annotated_frame
    
    def process_detections(self, frame):
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Leaf disease detection
        leaf_results = leaf_yolo.predict(source=frame, imgsz=416, conf=0.5, verbose=False)
        leaf_boxes = leaf_results[0].boxes.xyxy.cpu().numpy()
        
        for box in leaf_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Enlarge box
            box_w = x2 - x1
            box_h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            scale = 1.8
            new_w = box_w * scale
            new_h = box_h * scale
            new_x1 = int(max(cx - new_w / 2, 0))
            new_y1 = int(max(cy - new_h / 2, 0))
            new_x2 = int(min(cx + new_w / 2, w))
            new_y2 = int(min(cy + new_h / 2, h))
            
            crop = frame[new_y1:new_y2, new_x1:new_x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            
            input_tensor = resnet_transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                preds = resnet_model(input_tensor)
                pred_class = torch.argmax(preds, dim=1).item()
                disease_name = class_names[pred_class]
            
            color = (0, 255, 0) if disease_name == "Healthy" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, disease_name, (x1, max(0, y1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if disease_name != "Healthy":
                gps = gps_gen.get_next_location()
                logger.log_result("Leaf_Disease", disease_name, gps)
        
        # Weed detection
        weed_results = weed_yolo.predict(source=frame, imgsz=416, conf=0.5, verbose=False)
        weed_boxes = weed_results[0].boxes.xyxy.cpu().numpy()
        
        for box in weed_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(annotated_frame, "WEED", (x1, max(0, y1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            gps = gps_gen.get_next_location()
            logger.log_result("Weed", "WEED_DETECTED", gps)
        
        return annotated_frame

camera = CameraProcessor()
soil_monitor = SoilHealthMonitor()
soil_monitor.start()

# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_ip', methods=['POST'])
def set_ip():
    data = request.get_json()
    ip = data.get('ip', '')
    if not ip:
        return jsonify({"success": False, "message": "IP address required"})
    
    camera.set_phone_ip(ip)
    return jsonify({"success": True, "message": f"IP set to {ip}"})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    success, message = camera.start_camera()
    return jsonify({"success": success, "message": message})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    camera.stop_camera()
    return jsonify({"success": True, "message": "Camera stopped"})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    detections = logger.get_recent_detections()
    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)