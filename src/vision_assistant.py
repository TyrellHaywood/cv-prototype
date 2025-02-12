# dependencies
import cv2
import time

# functions
from object_detection import ObjectDetector
from edge_detection import EdgeDetector
from tts import SpeechProcessor
from overlay import TextOverlay

class VisionAssistant:
    def __init__(self):
        # initialize handlers
        self.detector = ObjectDetector()
        self.edge_detector = EdgeDetector()
        self.tts = SpeechProcessor()
        self.text_overlay = TextOverlay()
        self.last_edge_alert = 0

    def process_frame(self, frame, show_bboxes=True, show_overlay=True, tts_enabled=True):
        """Handles frame processing: object detection, edge detection, and speech feedback."""
        current_time = time.time()

        # Object detection
        detected_objects = self.detector.detect_objects(frame, show_bboxes)
        if tts_enabled:
            self.tts.handle_announcements(detected_objects)

        # Edge detection
        edges = self.edge_detector.detect_edges(frame)

        # Detect obstructions and update frame 
        frame = self.edge_detector.detect_obstructions(frame, edges, current_time, self.tts, show_overlay, tts_enabled)

        # draw text overlay
        frame = self.text_overlay.draw(frame)

        return frame  # Make sure frame is updated
    
    def update_text_overlay(self, message):
        """passes text message to overlay handler"""
        self.text_overlay.update_text(message)
