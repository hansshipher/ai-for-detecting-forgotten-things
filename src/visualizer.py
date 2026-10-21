import cv2
import config as cfg

class Visualizer:
    def draw(self, frame, persons, bags):
        for person_id, center in persons.items():
            cv2.circle(frame, center, 5, cfg.COLOR_PERSON, -1)
            cv2.putText(frame, f"P-{person_id}", (center[0] - 15, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.COLOR_PERSON, 1)

        for bag_id, bag_info in bags.items():
            x1, y1, x2, y2 = bag_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Bag #{bag_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return frame