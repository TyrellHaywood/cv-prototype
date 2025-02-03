from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import pyttsx3
import time
import os
import threading

class VisionAssistant:
    def __init__(self):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize and save models - using smaller models for better performance
        general_model = YOLO('yolov8n.pt')  # Using nano model instead of xlarge
        general_model.save('models/yolov8n.pt')
        
        # Load model from saved location - only using one model for better performance (for now)
        self.detector = YOLO('models/yolov8n.pt')
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Define critical objects for detection
        self.critical_objects = [
            'person', 'car', 'truck', 'bicycle', 'motorcycle',  # Road users
            'bench', 'chair', 'table', 'stop sign', 'traffic light',  # Obstacles and signals
        ]
        
        # Initialize last announcement time
        self.last_announcement = {}
        
        # Initialize processing flag
        self.is_processing = False
        
    def process_frame(self, frame):
        if self.is_processing:
            return frame
            
        self.is_processing = True
        
        results = self.detector(frame, verbose=False)[0]  # Get detection results

        detected_objects = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result
            class_name = results.names[int(class_id)]
            
            print(f"Raw detection: {class_name} ({conf:.2f})")

            if class_name in self.critical_objects and conf > 0.5:
                position = self._get_position(frame, x1, y1, x2, y2)
                detected_objects.append({
                    'name': class_name,
                    'confidence': conf,
                    'position': position
                })
                
                # **LOGGING DETECTED OBJECTS**
                print(f"Detected: {class_name} ({conf:.2f}) at {position}")

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name}: {conf:.2f}', 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        threading.Thread(target=self._generate_warnings, args=(detected_objects,), daemon=True).start()
        
        self.is_processing = False
        return frame

    
    def _get_position(self, frame, x1, y1, x2, y2):
        """Determine if object is left, center, or right in frame"""
        frame_width = frame.shape[1]
        center_x = (x1 + x2) / 2
        
        if center_x < frame_width / 3:
            return "left"
        elif center_x < 2 * frame_width / 3:
            return "center"
        else:
            return "right"
    
    def _generate_warnings(self, detected_objects):
        """Generate and speak appropriate warnings based on detected objects"""
        current_time = time.time()
        
        for obj in detected_objects:
            # Only announce each type of object once every 3 seconds
            if obj['name'] not in self.last_announcement or \
               current_time - self.last_announcement[obj['name']] > 3:
                
                message = f"{obj['name']} detected {obj['position']}"
                
                # Prioritize immediate dangers
                if obj['name'] in ['car', 'truck', 'motorcycle'] and obj['position'] == 'center':
                    message = f"WARNING! {message}"
                    self.engine.say(message)
                    self.engine.runAndWait()
                
                self.last_announcement[obj['name']] = current_time

def main():
    # Initialize the vision assistant
    assistant = VisionAssistant()
    
    # Initialize camera (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every 3rd frame
        if frame_count % 3 == 0:
            # Process frame
            processed_frame = assistant.process_frame(frame)
        else:
            processed_frame = frame
            
        frame_count += 1
        
        # Display frame
        cv2.imshow('Prototype', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()