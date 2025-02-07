import cv2
from vision_assistant import VisionAssistant

def main():
    assistant = VisionAssistant()
    cap = cv2.VideoCapture(0)

    frame_count = 0
    process_every_n_frames = 3

    # Setting states of visual feedback
    show_bounding_boxes = True
    show_obstruction_highlight = True
    tts_enabled = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % process_every_n_frames == 0:
            processed_frame = assistant.process_frame(frame, show_bounding_boxes)
        else:
            processed_frame = frame

        frame_count += 1
        cv2.imshow('prototype...', processed_frame)

        key = cv2.waitKey(1) & 0xFF # Define key press events

        if key == ord('q'):
            break
        elif key == ord('b'):  # Toggle Bounding Boxes
            show_bounding_boxes = not show_bounding_boxes
            print(f"Bounding Boxes {'ON' if show_bounding_boxes else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
