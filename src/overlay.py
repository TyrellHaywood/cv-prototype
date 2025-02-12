import cv2
import time

class TextOverlay:
    def __init__(self, timeout=3):
        self.display_text = ""  # Current text to display
        self.text_timeout = timeout  # Time (seconds) the text remains on screen
        self.last_text_update = 0  # Timestamp of the last update
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def update_text(self, message):
        """Updates the text message and resets the timer."""
        self.display_text = message
        self.last_text_update = time.time()

    def draw(self, frame):
        """Draws the text overlay on the given frame."""
        if self.display_text and (time.time() - self.last_text_update < self.text_timeout):
            cv2.putText(frame, self.display_text, (50, 50), 
                        self.font, 0.8, (28, 28, 28), 2, cv2.LINE_AA)
        return frame
