"""
Module: utils.py
Description: This module contains utility functions for the aircraft turnaround analytics application, including frame annotation and video processing helpers.
"""
import numpy as np
import cv2


def annotate_frame(
        active_tracks: list[dict],
        first_seen: dict,
        frame_bgr: np.ndarray,
        frame_idx: int,
        fps: float,
        x1: int,
        y1: int,
) -> list[dict]:
    """
    Annotate a single video frame with bounding boxes and labels based on active tracks.
    
    Args:
        active_tracks (list[dict]): List of active tracks from the tracker, where each track is a dictionary containing 'id', 'box', 'class', and 'lost
        first_seen (dict): Dictionary to track the first frame index when each object was seen.
        frame_bgr (np.ndarray): The current video frame in BGR format.
        frame_idx (int): The index of the current frame in the video.
        fps (float): Frames per second of the video.
        x1, y1 (int): Coordinates defining the Region of Interest (ROI) in the frame.


    Returns:
        frame_bgr (np.ndarray): The annotated video frame with bounding boxes and labels.
        first_seen (dict): Updated dictionary tracking the first frame index when each object was seen.
    """
    # Annotate the Original Frame
    for track in active_tracks:
        if track['lost_frames'] == 0: # Only draw if actively visible
            track_id = track['id']
            
            # Record birth frame if this is a new object
            if track_id not in first_seen:
                first_seen[track_id] = frame_idx
                
            # Calculate duration alive in seconds
            duration_sec = (frame_idx - first_seen[track_id]) / fps
            
            # Shift ROI box coordinates back to global frame coordinates
            box = track['box']
            global_xmin = int(box[0]) + x1
            global_ymin = int(box[1]) + y1
            global_xmax = int(box[2]) + x1
            global_ymax = int(box[3]) + y1
            
            # Formulate the label: "Truck #5 | 12.5s"
            label_text = f"{track['class'].capitalize()} #{track_id} | {duration_sec:.1f}s"
            
            # Draw Bounding Box (Color is BGR: Green)
            cv2.rectangle(frame_bgr, (global_xmin, global_ymin), (global_xmax, global_ymax), (0, 255, 0), 2)
            
            # Draw Text Background (for readability)
            (text_w, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_bgr, (global_xmin, global_ymin - 25), (global_xmin + text_w, global_ymin), (0, 255, 0), -1)
            
            # Draw Text (Color is BGR: Black)
            cv2.putText(frame_bgr, label_text, (global_xmin, global_ymin - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame_bgr, first_seen