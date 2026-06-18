"""
Module: tracker.py
Description: This module implements a simple IoU-based tracking algorithm to maintain object identities across video frames
"""

# Core Geometric Math
def compute_iou(boxA, boxB):
    # Extract coordinates
    xA = max(boxA[0], boxB[0]) # xA is the maximum of the left x-coordinates of both boxes
    yA = max(boxA[1], boxB[1]) # yA is the maximum of the top y-coordinates of both boxes
    xB = min(boxA[2], boxB[2]) # xB is the minimum of the right x-coordinates of both boxes
    yB = min(boxA[3], boxB[3]) # yB is the minimum of the bottom y-coordinates of both boxes

    # Calculate intersection area
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0  # No overlap

    # Calculate union
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou

# Stateful Tracker Architecture
class GreedyIoUTracker:
    """A simple IoU-based tracker that maintains object identities across frames."""
    def __init__(self, iou_threshold=0.3, max_lost_frames=5):
        """Initialize the tracker with IoU threshold and maximum lost frames."""
        self.tracks = [] # will hold dicts: {'id': int, 'box': list, 'class': str, 'lost_frames': int}
        self.next_id = 1
        self.iou_threshold = iou_threshold

        self.max_lost = max_lost_frames # Crucial for occlusions: If a worker walks behind a truck, we wait N frames before deleting their ID

    def update(self, new_boxes, new_classes):
        """
        Update the tracker with new detections. This method attempts to match existing tracks to new detections based on IoU.
        """
        updated_tracks = []
        matched_new_boxes = set()

        # Phase A: Attempt to match existing tracks to new detections based on IoU
        for track in self.tracks:
            best_iou = 0
            best_match_idx = -1

            for i, box in enumerate(new_boxes):
                if i in matched_new_boxes:
                    continue  # Skip already matched boxes

                iou = compute_iou(track['box'], box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

            if best_iou > self.iou_threshold:
                # SUCCESS: Track maintained, update coordinates and reset lost frame count
                track['box'] = new_boxes[best_match_idx]
                track['lost_frames'] = 0
                updated_tracks.append(track)
                matched_new_boxes.add(best_match_idx)
            else:
                # FAILURE: Track lost, increment lost frame count
                track['lost_frames'] += 1
                if track['lost_frames'] <= self.max_lost:
                    updated_tracks.append(track)  # Keep the track for now in case it reappears

        # Phase B: Register unmatched detections as new tracks
        for i, box in enumerate(new_boxes):
            if i not in matched_new_boxes:
                new_track = {
                    'id': self.next_id,
                    'box': box,
                    'class': new_classes[i],
                    'lost_frames': 0
                }
                updated_tracks.append(new_track)
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks

    def get_active_tracks(self):
        """Return a list of currently active tracks (those that haven't been lost for too long)."""
        return [track for track in self.tracks if track['lost_frames'] <= self.max_lost]