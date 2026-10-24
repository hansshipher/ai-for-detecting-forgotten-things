import cv2
import config as cfg

class Visualizer:
    def __init__(self):
        self.status_colors = {
            'SAFE': cfg.COLOR_SAFE,
            'WARNING': cfg.COLOR_WARNING,
            'ALARM': cfg.COLOR_ALARM,
        }
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

    def draw_status(self, frame, persons, tracked_bags):
        for person_id, center in persons.items():
            cv2.circle(frame, center, 5, cfg.COLOR_PERSON, -1)
            cv2.putText(frame, f"P-{person_id}", (center[0] - 15, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg.COLOR_PERSON, 1)

        for bag_id, bag_data in tracked_bags.items():
            status = bag_data['status']
            elapsed = bag_data['elapsed']
            x1, y1, x2, y2 = bag_data['bbox']
            color = self.status_colors.get(status, cfg.COLOR_SAFE)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if status in ('WARNING', 'ALARM'):
                cv2.circle(frame, center, cfg.SAFE_RADIUS_PX, color, 1, lineType=cv2.LINE_AA)

            thickness = 3 if status == 'ALARM' else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if status == 'SAFE':
                label = f"Bag #{bag_id} [OK]"
            elif status == 'WARNING':
                label = f"Bag #{bag_id} [{elapsed:.1f}s]"
            else:
                label = f"ALARM! Bag #{bag_id} [{elapsed:.1f}s]"

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            _, frame_w = frame.shape[:2]
            pad = 10
            x1_label = max(0, min(x1, frame_w - w - pad))

            cv2.rectangle(frame, (x1_label, y1 - 20), (x1_label + w + pad, y1), color, -1)
            text_color = (0, 0, 0) if status == 'WARNING' else cfg.COLOR_TEXT
            cv2.putText(frame, label, (x1_label + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        return frame


if __name__ == "__main__":
    import sys, os
    import numpy as np
    import cv2
    sys.path.insert(0, os.path.dirname(__file__))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    persons = {1: (500, 240)}
    tracked_bags = {
        7: {'status': 'WARNING', 'elapsed': 2.3, 'bbox': (100, 200, 160, 280)},
        9: {'status': 'ALARM',   'elapsed': 6.0, 'bbox': (560, 300, 620, 380)},
    }

    vis = Visualizer()
    out = vis.draw_status(frame, persons, tracked_bags)
    cv2.imwrite("src/_viz_check.png", out)
    print("saved src/_viz_check.png")