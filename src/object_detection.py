from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.critical_objects = {'person', 'car', 'chair', 'stair'}

    def detect_objects(self, frame, show_bboxes=True):
        """Runs YOLOv8 object detection and returns detected objects."""
        results = self.model.track(frame, verbose=False)[0]
        detected_objects = []

        for result in results.boxes.data.tolist():
            # some results might not have track_id, so we handle it safely
            if len(result) == 7:
                x1, y1, x2, y2, conf, class_id, track_id = result
            else:
                x1, y1, x2, y2, conf, class_id = result
                track_id = None  # assign None when track_id is missing

            class_name = results.names[int(class_id)]

            if class_name in self.critical_objects and conf > 0.5:
                detected_objects.append({
                    'name': class_name, 
                    'position': self._get_position(frame, x1, y1, x2, y2),
                    'id': track_id
                    })

            # Draw bounding boxes only if enabled
            if show_bboxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'{class_name} ({int(track_id)}): {conf:.2f}' if track_id is not None else f'{class_name}: {conf:.2f}'
                cv2.putText(frame, label, 
                    (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return detected_objects

    def _get_position(self, frame, x1, y1, x2, y2):
        """Determines object position relative to frame (left, center, right)."""
        center_x = (x1 + x2) / 2
        frame_width = frame.shape[1]

        if center_x < frame_width / 3:
            return "left"
        elif center_x < 2 * frame_width / 3:
            return "center"
        else:
            return "right"
