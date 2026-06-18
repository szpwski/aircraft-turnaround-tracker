import pandas as pd
from src.etl import TracksDataProcessor, TracksDataColumns


def test_tracks_data_processor_log_transform_and_kpis():
    proc = TracksDataProcessor()

    # Frame 10: both a truck and a person appear
    frame10_tracks = [
        {'id': 1, 'class': 'truck', 'first_seen_frame': 10, 'last_seen_frame': 10, 'lost_frames': 0},
        {'id': 2, 'class': 'person', 'first_seen_frame': 10, 'last_seen_frame': 10, 'lost_frames': 0},
    ]
    proc.log_state(frame10_tracks, frame_idx=10)

    # Frame 30: truck still present
    frame30_tracks = [
        {'id': 1, 'class': 'truck', 'first_seen_frame': 10, 'last_seen_frame': 30, 'lost_frames': 0},
    ]
    proc.log_state(frame30_tracks, frame_idx=30)

    # Transform using fps=10 -> seconds conversion: 10->1s, 30->3s
    proc.transform_data(fps=10.0)
    df = proc.get_dataframe()

    # Should have two unique vehicle entries
    assert df.shape[0] == 2

    # Validate truck timing and duration
    truck_row = df[df[TracksDataColumns.VEHICLE_ID.value] == 'Truck #1'].iloc[0]
    assert truck_row[TracksDataColumns.START_SEC.value] == 1.0
    assert truck_row[TracksDataColumns.END_SEC.value] == 3.0
    assert truck_row[TracksDataColumns.DURATION_SEC.value] == 2.0

    # Plot start/end columns should be datetime-like
    assert pd.api.types.is_datetime64_any_dtype(df[TracksDataColumns.PLOT_START.value])
    assert pd.api.types.is_datetime64_any_dtype(df[TracksDataColumns.PLOT_END.value])

    # KPIs
    kpis = proc.get_kpis()
    assert kpis['total_vehicles'] == 1
    assert kpis['total_personnel'] == 1
    assert kpis['max_duration'] == 2.0
