import pyttsx3
import threading
from queue import Queue
import time

class SpeechProcessor:
    def __init__(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.thread.start()
        self.last_alert = 0
        self.tracked_objects = {}

    def _tts_worker(self):
        """Continuously processes messages in the TTS queue."""
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        engine.setProperty('voice', engine.getProperty('voices')[1].id)  # Use female voice

        while True:
            message = self.queue.get()
            engine.say(message)
            engine.runAndWait()
            self.queue.task_done()

    def queue_message(self, message):
        """Queues a message for speech output."""
        self.queue.put(message)

    def handle_announcements(self, detected_objects):
        """Tracks detected objects and announces when needed."""
        current_time = time.time()
        disappeared_objects = set(self.tracked_objects.keys())

        for obj in detected_objects:
            obj_key = f"{obj['name']}_{obj['position']}"

            if obj_key in self.tracked_objects:
                disappeared_objects.discard(obj_key)
            else:
                if current_time - self.tracked_objects.get(obj_key, 0) > 5:
                    self.queue_message(f"{obj['name']} detected {obj['position']}")

            self.tracked_objects[obj_key] = current_time

        for obj_key in disappeared_objects:
            if current_time - self.tracked_objects[obj_key] > 5:
                del self.tracked_objects[obj_key]
