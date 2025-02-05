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
        self.last_announcement = {}
        self.is_processing = False

        self.edge_threshold = 10000  # Threshold for strong edge detection
        self.last_edge_alert = 0  # Track last edge alert time

        
    def _tts_worker(self):
        """Worker thread for text-to-speech processing"""
        # Initialize TTS engine once for the thread
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # Use female voice
        
        while True:
            # Wait for messages in the queue
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
        detected_objects = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result
            class_name = results.names[int(class_id)]
            
            # Filter for critical objects with high confidence
            if class_name in self.critical_objects and conf > 0.5:
                position = self._get_position(frame, x1, y1, x2, y2)
                detected_objects.append({
                    'name': class_name,
                    'confidence': conf,
                    'position': position
                })
                
                # Draw detection visualization
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name}: {conf:.2f}', 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Apply edge detection
        edges = self._detect_edges(frame)
        self._detect_obstructions(edges)
        frame = cv2.addWeighted(frame, 0.8, edges, 0.2, 0)

        # Generate warnings for detected objects
        self._generate_warnings(detected_objects)
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
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast
        equalized = cv2.equalizeHist(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        # Canny edge detection with optimized thresholds
        canny_edges = cv2.Canny(blurred, 75, 175)

        # Laplacian edge detection (captures intensity changes)
        laplacian_edges = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian_edges = np.uint8(np.absolute(laplacian_edges))

        # Sobel edge detection (captures directional edges)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = cv2.bitwise_or(np.uint8(np.absolute(sobel_x)), np.uint8(np.absolute(sobel_y)))

        # Combine different edge maps
        combined_edges = cv2.bitwise_or(canny_edges, laplacian_edges)
        combined_edges = cv2.bitwise_or(combined_edges, sobel_edges)

        # Convert edges to 3-channel image for overlay
        edges_colored = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)

        return edges_colored

    def _detect_obstructions(self, edges):
        current_time = time.time()
        edge_strength = np.sum(edges) / 255
        if edge_strength > self.edge_threshold and current_time - self.last_edge_alert > 3:
            self.tts_queue.put("Obstruction ahead")
            self.last_edge_alert = current_time

    def _generate_warnings(self, detected_objects):
        """Generate and queue speech warnings for detected objects"""
        current_time = time.time()
        
        for obj in detected_objects:
            # Only announce each object type every 3 seconds
            if obj['name'] not in self.last_announcement or \
               current_time - self.last_announcement[obj['name']] > 3:
                
                message = f"{obj['name']} detected {obj['position']}"
                
                # Add extra warning for dangerous objects in center
                if obj['name'] in ['car', 'truck', 'motorcycle'] and obj['position'] == 'center':
                    message = f"WARNING! {message}"
                
                # Queue message for TTS thread
                self.tts_queue.put(message)
                self.last_announcement[obj['name']] = current_time

def main():
    # Initialize system
    assistant = VisionAssistant()
    cap = cv2.VideoCapture(0)
    
    # Lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Frame processing control
    frame_count = 0
    process_every_n_frames = 3  # Only process every 3rd frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame if it's time
        if frame_count % process_every_n_frames == 0:
            processed_frame = assistant.process_frame(frame)
        else:
            processed_frame = frame
            
        frame_count += 1
        
        # Display output
        cv2.imshow('Prototype', processed_frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()