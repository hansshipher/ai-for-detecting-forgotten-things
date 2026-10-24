import math
import config as cfg

class AbObjDetector:
    def __init__(self, max_missing_frames=45, spatial_threshold=50,
                 person_link_radius=80, person_max_missing=45):
        self.tracked_bags = {}
        self.max_missing_frames = max_missing_frames      
        self.spatial_threshold = spatial_threshold        
        self.person_link_radius = person_link_radius      
        self.person_max_missing = person_max_missing      
        self.persons_state = {}                           
        self._next_lid = 0

    def _link_persons(self, persons):
        used = set()
        for track_id, pos in persons.items():
            for lid, st in self.persons_state.items():
                if st['track_id'] == track_id:
                    st['pos'] = pos
                    st['missing'] = 0
                    used.add(lid)
                    break

        for track_id, pos in persons.items():
            if any(st['track_id'] == track_id for st in self.persons_state.values()):
                continue
            best_lid, best_d = None, float('inf')
            for lid, st in self.persons_state.items():
                if lid in used or st['missing'] == 0:
                    continue
                d = math.hypot(pos[0] - st['pos'][0], pos[1] - st['pos'][1])
                if d < self.person_link_radius and d < best_d:
                    best_d, best_lid = d, lid
            if best_lid is not None:
                self.persons_state[best_lid]['track_id'] = track_id
                self.persons_state[best_lid]['pos'] = pos
                self.persons_state[best_lid]['missing'] = 0
                used.add(best_lid)
            else:
                self.persons_state[self._next_lid] = {'track_id': track_id, 'pos': pos, 'missing': 0}
                used.add(self._next_lid)
                self._next_lid += 1

        for lid in list(self.persons_state.keys()):
            if lid not in used:
                self.persons_state[lid]['missing'] += 1
                if self.persons_state[lid]['missing'] > self.person_max_missing:
                    del self.persons_state[lid]

    def update(self, persons, bags, video_time):
        self._link_persons(persons)
        matched_bag_ids = set()

        for bag_id, bag_info in bags.items():
            bag_center = bag_info['center']
            bag_bbox = bag_info['bbox']
            target_state_key = bag_id

            if bag_id not in self.tracked_bags:
                best_match_id, min_dist = None, float('inf')
                for existing_id, state in self.tracked_bags.items():
                    old_center = ((state['bbox'][0] + state['bbox'][2]) // 2,
                                  (state['bbox'][1] + state['bbox'][3]) // 2)
                    d = math.hypot(bag_center[0] - old_center[0], bag_center[1] - old_center[1])
                    if d < self.spatial_threshold and d < min_dist:
                        min_dist, best_match_id = d, existing_id

                if best_match_id is not None:
                    print(f"[INFO] ID Switch detected: Bag #{best_match_id} -> New ID #{bag_id}")
                    self.tracked_bags[bag_id] = self.tracked_bags.pop(best_match_id)
                    target_state_key = bag_id
                else:
                    owner_lid, min_d = None, float('inf')
                    for lid, st in self.persons_state.items():
                        if st['missing'] != 0:
                            continue
                        d = math.hypot(bag_center[0] - st['pos'][0], bag_center[1] - st['pos'][1])
                        if d < min_d:
                            min_d, owner_lid = d, lid
                    self.tracked_bags[bag_id] = {
                        'status': 'SAFE', 'owner_lid': owner_lid,
                        'start_time': None, 'elapsed': 0.0,
                        'bbox': bag_bbox, 'missing_count': 0
                    }
                    target_state_key = bag_id

            state = self.tracked_bags[target_state_key]
            state['bbox'] = bag_bbox
            state['missing_count'] = 0
            matched_bag_ids.add(target_state_key)

            owner_near = False
            olid = state['owner_lid']
            if olid is not None and olid in self.persons_state:
                pst = self.persons_state[olid]
                if pst['missing'] == 0 and math.hypot(
                    bag_center[0] - pst['pos'][0], bag_center[1] - pst['pos'][1]
                ) < cfg.SAFE_RADIUS_PX:
                    owner_near = True

            if owner_near:
                state['status'] = 'SAFE'
                state['start_time'] = None
                state['elapsed'] = 0.0
            else:
                if state['start_time'] is None:
                    state['start_time'] = video_time
                state['elapsed'] = video_time - state['start_time']
                state['status'] = 'ALARM' if state['elapsed'] >= cfg.ALARM_DELAY_SEC else 'WARNING'

        for b_id in list(self.tracked_bags.keys()):
            if b_id not in matched_bag_ids:
                self.tracked_bags[b_id]['missing_count'] += 1
                if self.tracked_bags[b_id]['missing_count'] > self.max_missing_frames:
                    del self.tracked_bags[b_id]

        return self.tracked_bags