from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import pyttsx3
import time
import os
import threading
from queue import Queue

class VisionAssistant:
    def __init__(self):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize and save models - using smaller models for better performance
        general_model = YOLO('yolov8n.pt')  # Using nano model for speed
        general_model.save('models/yolov8n.pt')
        
        # Load model from saved location
        self.detector = YOLO('models/yolov8n.pt')
        
        # Initialize TTS queue and worker thread
        self.tts_queue = Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        # Define critical objects for detection
        self.critical_objects = [
            # People
            'person',

            # Vehicles
            'car', 'truck', 'bicycle', 'motorcycle', 'bus',

            # Common Furniture
            'chair', 'table', 'bench', 'couch', 'bed', 'desk', 'cabinet', 'shelf',

            # Obstructions
            'stop sign', 'traffic light', 'wall', 'pole', 'trash can',

            # Changes in Elevation
            'stair', 'ramp', 'curb', 'escalator', 'elevator'
        ]
        
        # Track last announcements to prevent spam
        self.last_announcement = {}  # Object name -> timestamp
        self.detected_objects = {}  # Tracks objects with positions to avoid repeats
        self.is_processing = False


        self.edge_threshold = 10000  # Threshold for strong edge detection
        self.last_edge_alert = 0  # Track last edge alert time
        self.edge_alert_counter = 0
        self.edge_alert_threshold = 3  # Only alert if edges are detected for 3+ frames consecutively


        self.tracked_objects = {}  # Tracks object presence and last seen time

    def _generate_warnings(self, detected_objects):
        """Generate and queue speech warnings for detected objects, preventing unnecessary repeats."""
        current_time = time.time()
        disappeared_objects = set(self.tracked_objects.keys())  # Assume all old objects disappeared

        for obj in detected_objects:
            obj_name, obj_position = obj['name'], obj['position']
            obj_key = f"{obj_name}_{obj_position}"

            # Object is still in view, update last seen timestamp
            if obj_key in self.tracked_objects:
                disappeared_objects.discard(obj_key)  # It's still visible
            else:
                # Announce only if the object was gone for 5+ seconds before reappearing
                last_seen_time = self.tracked_objects.get(obj_key, 0)
                if current_time - last_seen_time > 5:
                    message = f"{obj_name} detected {obj_position}"
                    if obj_name in ['car', 'truck', 'motorcycle'] and obj_position == 'center':
                        message = f"WARNING! {message}"

                    self.tts_queue.put(message)

            # Update object's last seen time
            self.tracked_objects[obj_key] = current_time

        # Forget objects that have disappeared for 5+ seconds
        for obj_key in disappeared_objects:
            if current_time - self.tracked_objects[obj_key] > 5:
                del self.tracked_objects[obj_key]

    def _tts_worker(self):
        """Worker thread for text-to-speech processing"""
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # Use female voice
        
        while True:
            message = self.tts_queue.get()
            engine.say(message)
            engine.runAndWait()
            self.tts_queue.task_done()
    
    def process_frame(self, frame):
        """Process a single frame with object detection"""
        if self.is_processing:
            return frame
            
        self.is_processing = True
        
        # Run YOLO detection
        results = self.detector(frame, verbose=False)[0]
        current_time = time.time()
        detected_objects = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result
            class_name = results.names[int(class_id)]
            
            if class_name in self.critical_objects and conf > 0.5:
                position = self._get_position(frame, x1, y1, x2, y2)
                object_key = f"{class_name}_{position}"
                
                detected_objects.append({'name': class_name, 'position': position})
                
                # Check if this is a new detection or if enough time has passed since last detection
                self._generate_warnings(detected_objects)
                
                # Draw detection visualization
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name}: {conf:.2f}', 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Apply edge detection
        edges = self._detect_edges(frame)
        self._detect_obstructions(edges, current_time)
        frame = cv2.addWeighted(frame, 0.8, edges, 0.2, 0)
        
        self.is_processing = False
        return frame
    
    def _get_position(self, frame, x1, y1, x2, y2):
        """Determine relative position of object in frame"""
        frame_width = frame.shape[1]
        center_x = (x1 + x2) / 2
        
        if center_x < frame_width / 3:
            return "left"
        elif center_x < 2 * frame_width / 3:
            return "center"
        else:
            return "right"
    
    def _detect_edges(self, frame):
        """Detect edges using multiple edge detection techniques"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        canny_edges = cv2.Canny(blurred, 75, 175)
        laplacian_edges = np.uint8(np.absolute(cv2.Laplacian(blurred, cv2.CV_64F)))
        sobel_x = np.uint8(np.absolute(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)))
        sobel_y = np.uint8(np.absolute(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)))
        
        combined_edges = cv2.bitwise_or(canny_edges, laplacian_edges)
        combined_edges = cv2.bitwise_or(combined_edges, sobel_x)
        combined_edges = cv2.bitwise_or(combined_edges, sobel_y)
        return cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)
    
    def _detect_obstructions(self, edges, current_time):
        """Detect strong edges and prevent repeated announcements"""
        edge_strength = np.sum(edges) / 255
        
        if edge_strength > self.edge_threshold:
            self.edge_alert_counter += 1
        else:
            self.edge_alert_counter = 0  # Reset if obstruction disappears

        # Announce only if obstruction is seen for multiple consecutive frames
        
        if self.edge_alert_counter >= self.edge_alert_threshold:
            red_overlay = np.zeros_like(edges, dtype=np.uint8) # Create a red overlay for detected obstructions
            red_overlay[:] = (0, 0, 255)  # Red color (BGR format)
            edges = cv2.addWeighted(edges, 0.6, red_overlay, 0.4, 0)  # Blend with 40% transparency

            if current_time - self.last_edge_alert > 5:
                self.tts_queue.put("Obstruction ahead")
                self.last_edge_alert = current_time
                self.edge_alert_counter = 0  # Reset after announcement

def main():
    assistant = VisionAssistant()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % process_every_n_frames == 0:
            processed_frame = assistant.process_frame(frame)
        else:
            processed_frame = frame
            
        frame_count += 1
        
        cv2.imshow('Prototype', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()