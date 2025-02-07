import cv2
from vision_assistant import VisionAssistant

def main():
    assistant = VisionAssistant()
    cap = cv2.VideoCapture(0)

    frame_count = 0
    process_every_n_frames = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % process_every_n_frames == 0:
            processed_frame = assistant.process_frame(frame)
        else:
            processed_frame = frame

        frame_count += 1
        cv2.imshow('Vision Assistant', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
