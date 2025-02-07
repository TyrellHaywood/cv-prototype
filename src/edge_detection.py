import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, edge_threshold=10000):
        self.edge_threshold = edge_threshold
        self.edge_alert_counter = 0
        self.edge_alert_threshold = 3  # Consecutive frames before alert

    def detect_edges(self, frame):
        """Applies multiple edge detection techniques and combines them."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_edges = cv2.Canny(blurred, 75, 175)

        return cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

    def detect_obstructions(self, frame, edges, current_time, tts, show_overlay=True, tts_enabled=True):
        """Detects strong edges and provides a bright red overlay for obstructions."""
        edge_strength = np.sum(edges) / 255

        if edge_strength > self.edge_threshold:
            self.edge_alert_counter += 1
        else:
            self.edge_alert_counter = 0  # Reset if obstruction disappears

        if self.edge_alert_counter >= self.edge_alert_threshold:

            # Show edge overlay only if if enabled
            if show_overlay:
                red_overlay = np.zeros_like(frame, dtype=np.uint8)
                red_overlay[:] = (0, 0, 255)
                frame = cv2.addWeighted(frame, 0.6, red_overlay, 0.4, 0)

            # # Apply only on strong edges
            # mask = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY) > 50
            # frame[mask] = cv2.addWeighted(frame, 0.4, red_overlay, 0.6, 0)[mask]

            if (current_time - tts.last_alert > 5) and tts_enabled:
                tts.queue_message("Obstruction ahead")
                tts.last_alert = current_time
                self.edge_alert_counter = 0  # Reset

        return frame # Return the modified frame
