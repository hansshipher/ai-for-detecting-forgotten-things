import cv2
import sys
import config as cfg
from tracker import ObjTracker
from visualizer import Visualizer

def main():
    if not cfg.VIDEO_PATH.exists():
        print(f"[ERROR] Cannot found video: {cfg.VIDEO_PATH}")
        sys.exit(1)
    
    tracker = ObjTracker()
    visualizer = Visualizer()
    
    cap = cv2.VideoCapture(str(cfg.VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video")
        sys.exit(1)
    
    print("[INFO] Test launch")

    persons = {}
    bags = {}
    
    frame_count = 0
    SKIP_FRAMES = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if frame_count % SKIP_FRAMES == 0:
            persons, bags = tracker.process_frame(frame)

        output_frame = visualizer.draw(frame, persons, bags)
        
        cv2.imshow("Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()