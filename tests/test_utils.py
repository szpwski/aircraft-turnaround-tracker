import numpy as np
from src.utils import annotate_frame


def test_annotate_frame_updates_first_seen_and_draws():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    active_tracks = [{
        'id': 5,
        'box': [10, 10, 20, 20],
        'class': 'truck',
        'lost_frames': 0
    }]
    first_seen = {}
    frame_idx = 100
    fps = 25.0
    x1, y1 = 0, 0

    annotated_frame, updated_first_seen = annotate_frame(active_tracks, first_seen, frame.copy(), frame_idx, fps, x1, y1)

    assert 5 in updated_first_seen
    assert updated_first_seen[5] == frame_idx
    # Some pixels should have changed from black due to drawing
    assert np.any(annotated_frame != 0)

