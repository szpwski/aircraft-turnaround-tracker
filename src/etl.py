
"""
Module: etl.py
Description: This module defines the TracksDataProcessor class, which processes raw tracking data to calculate start time, end time, and duration for each tracked object, and structures it for analysis.
"""
from enum import StrEnum 
import pandas as pd
from datetime import datetime


class TracksDataColumns(StrEnum):
    VEHICLE_ID = "Vehicle ID"
    FRAME_IDX = "Frame Index"
    CATEGORY = "Category"
    START_FRAME = "Start Frame"
    END_FRAME = "End Frame"
    START_SEC = "Start Sec"
    END_SEC = "End Sec"
    DURATION_SEC = "Duration (s)"
    PLOT_START = "Plot Start"
    PLOT_END = "Plot End"


class TracksDataProcessor:
    """
    Class to process track data and convert it into a structured format for analysis.
    """
    def __init__(self):
        """Initialize the TracksDataProcessor with an empty DataFrame."""
        self.data = pd.DataFrame()

    def log_state(self, tracks: list[dict], frame_idx: int):
        """
        Log the state of tracks at a given frame index. This method can be used for debugging and understanding the tracking process.

        Args:
            tracks (list[dict]): List of track dictionaries containing 'id', 'class', 'first_seen_frame', and 'last_seen_frame'.
            frame_idx (int): The current frame index being processed.
        """
        for track in tracks:
            if track["lost_frames"] == 0:  # Only log active tracks
                self.data = pd.concat([self.data, pd.DataFrame([{
                    TracksDataColumns.FRAME_IDX: frame_idx,
                    TracksDataColumns.VEHICLE_ID: f"{track['class'].capitalize()} #{track['id']}",
                    TracksDataColumns.CATEGORY: track['class']
                }])], axis=0)

    def transform_data(self, fps: float):
        """
        Transform the extracted data to calculate start time, end time, and duration for each track.
        This method updates the DataFrame with 'Start Sec', 'End Sec', and 'Duration (s)' columns.
        
        Args:
            fps (float): Frames per second of the video, used to convert frame indices to seconds.
        """
        service_times = self.data.groupby(
            TracksDataColumns.VEHICLE_ID.value
        )[TracksDataColumns.FRAME_IDX.value]\
            .agg(['min', 'max']).reset_index()
        service_times.columns = [TracksDataColumns.VEHICLE_ID.value, TracksDataColumns.START_FRAME.value, TracksDataColumns.END_FRAME.value]

        self.data = self.data.merge(service_times, on=TracksDataColumns.VEHICLE_ID.value)
        self.data.drop_duplicates(subset=[TracksDataColumns.VEHICLE_ID.value], inplace=True)
        self.data[TracksDataColumns.START_SEC.value] = self.data[TracksDataColumns.START_FRAME.value] / fps
        self.data[TracksDataColumns.END_SEC.value] = self.data[TracksDataColumns.END_FRAME.value] / fps
        self.data[TracksDataColumns.DURATION_SEC.value] = self.data[TracksDataColumns.END_SEC.value] - self.data[TracksDataColumns.START_SEC.value]

        # Create a dummy midnight baseline
        base_time = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Use vectorized pd.to_timedelta to instantly convert the entire column
        # (We create new columns 'plot_start' and 'plot_end' so we don't ruin the raw numeric seconds for your CSV export)
        self.data[TracksDataColumns.PLOT_START.value] = base_time + pd.to_timedelta(self.data[TracksDataColumns.START_SEC.value], unit='s')
        self.data[TracksDataColumns.PLOT_END.value] = base_time + pd.to_timedelta(self.data[TracksDataColumns.END_SEC.value], unit='s')

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the processed track data as a DataFrame.

        Returns:
            DataFrame containing processed track data.
        """
        return self.data

    def get_kpis(self) -> dict:
        """
        Calculate key performance indicators (KPIs) from the processed track data.

        Returns:
            Dictionary containing KPIs such as total vehicles, total personnel, and longest service time.
        """
        total_vehicles = len(self.data[self.data[TracksDataColumns.CATEGORY.value] == "truck"])
        total_personnel = len(self.data[self.data[TracksDataColumns.CATEGORY.value] == "person"])
        max_duration = self.data[TracksDataColumns.DURATION_SEC.value].max()
        return {
            "total_vehicles": total_vehicles,
            "total_personnel": total_personnel,
            "max_duration": max_duration
        }