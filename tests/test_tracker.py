from src.tracker import compute_iou, GreedyIoUTracker
import pytest
import numpy as np


def test_compute_iou_no_overlap():
    boxA = [0, 0, 10, 10]
    boxB = [10, 10, 20, 20]
    assert compute_iou(boxA, boxB) == 0.0


def test_compute_iou_partial_overlap():
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    iou = compute_iou(boxA, boxB)
    # intersection area = 25, union = 175 -> 25/175 ~= 0.142857
    assert pytest.approx(iou, rel=1e-3) == 25 / 175


def test_greedy_iou_tracker_register_and_match():
    tracker = GreedyIoUTracker(iou_threshold=0.1)
    boxes = [[0, 0, 10, 10]]
    classes = ["truck"]

    tracks = tracker.update(boxes, classes)
    assert len(tracks) == 1
    first_id = tracks[0]["id"]

    # Slightly moved box should match existing track
    new_boxes = [[1, 1, 11, 11]]
    new_classes = ["truck"]
    tracks2 = tracker.update(new_boxes, new_classes)
    assert len(tracks2) == 1
    assert tracks2[0]["id"] == first_id


def test_greedy_iou_tracker_lost_frames_removal():
    tracker = GreedyIoUTracker(iou_threshold=0.1, max_lost_frames=2)
    boxes = [[0, 0, 10, 10]]
    classes = ["truck"]
    tracker.update(boxes, classes)

    # Now simulate missing detections for several frames
    tracker.update([], [])  # lost_frames -> 1 (kept)
    assert len(tracker.tracks) == 1
    tracker.update([], [])  # lost_frames -> 2 (kept)
    assert len(tracker.tracks) == 1
    tracker.update([], [])  # lost_frames -> 3 (removed)
    assert len(tracker.tracks) == 0
