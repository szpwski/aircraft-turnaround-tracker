import streamlit as st
import cv2
import tempfile
import os
import subprocess
import plotly.express as px

from src.model import ObjectDetector
from src.tracker import GreedyIoUTracker
from src.utils import annotate_frame
from src.etl import TracksDataProcessor, TracksDataColumns


MAX_TIME_GUARDRAIL_SECONDS = 10  # Limit processing to first 15 seconds of video for demo purposes
INFERENCE_INTERVAL = 15 # Process every 15th frame to reduce load on the model and tracker
# --- Page Config & UI ---
st.set_page_config(page_title="Turnaround Analytics", layout="wide")
st.title("🛫 Post-Event Turnaround Analytics")
st.markdown("Analyze aircraft ground operations from live webcam streams.")

# --- Sidebar: Pipeline Settings ---
st.sidebar.header("⚙️ Pipeline Settings")
st.sidebar.markdown(
    """
    *Use these controls to fine-tune the tracking engine:*
    * **Confidence:** Minimum certainty required to detect a vehicle.
    * **IoU:** Overlap required to link a vehicle between frames.
    * **Lost Frames:** How long to "remember" a vehicle if it gets hidden behind a wing.
    """
)

# Hyperparameter Controls
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=1.00, value=0.95, step=0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", min_value=0.05, max_value=1.00, value=0.25, step=0.05)
max_lost_frames = st.sidebar.number_input("Max Lost Frames", min_value=1, max_value=300, value=60, step=5)

# --- Initialize Model and Tracker ---
@st.cache_resource
def load_ai_models():
    # Cache the model so it doesn't reload into memory on every UI click
    return ObjectDetector(device='cuda', confidence_threshold=conf_threshold)
model = load_ai_models()
model.confidence_threshold = conf_threshold  # Update model confidence threshold based on sidebar input
tracker = GreedyIoUTracker(iou_threshold=iou_threshold, max_lost_frames=max_lost_frames)
processor = TracksDataProcessor()
first_seen = {}  # To track the first frame index when each object was seen

# --- Sidebar Inputs ---
st.sidebar.markdown("---")
st.sidebar.header("🎬 Data Source")
# Replaced YouTube with a robust Local File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Turnaround Video", type=["mp4", "avi", "mov"])

if st.sidebar.button("▶️ Run Analysis"):
    if not uploaded_file:
        st.warning("Please upload a valid video file.")
    elif uploaded_file.size > 100 * 1024 * 1024:
        st.error("File too large")
    else:
        st.info("Video loaded. Running processing...")

        # Save uploaded file to a temporary location for OpenCV to read
        temp_dir = tempfile.gettempdir()
        input_video_path = os.path.join(temp_dir, "input_upload.mp4")
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Open CV Setup
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps: fps = 30  # Default to 30 if unable to get FPS

        # Setup temporary files for video generation
        temp_dir = tempfile.gettempdir()
        raw_avi_path = os.path.join(temp_dir, "raw_output.avi")
        final_mp4_path = os.path.join(temp_dir, "final_web.mp4")

        out = None
        width, height = 0, 0
        max_frames = int(MAX_TIME_GUARDRAIL_SECONDS * fps) # Guardrail to limit processing
        frame_count = 0

        # Streamlit progress bar
        progress_bar = st.progress(0)

        # Processing loop
        current_tracks = [] # Stores the active bounding boxes and IDs
        while frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                break

            # Initialize video writer on first frame
            if out is None:
                height, width = frame.shape[:2]
                # Setup ROI based on actual frame dimensions
                y1, y2 = int(height * 0.2), int(height * 0.9)
                x1, x2 = int(width * 0.1), int(width * 0.9)
                out = cv2.VideoWriter(raw_avi_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

            # Crop to ROI and run inference
            roi_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]

            # THE INFERENCE GATE
            if frame_count % INFERENCE_INTERVAL == 0:
                # Inference on model
                boxes, labels, scores = model.detect(roi_rgb)

                # Update tracker with new detections
                tracks = tracker.update(boxes, labels)

            # Annotate frame with active tracks
            frame_bgr, first_seen = annotate_frame(tracks, first_seen, frame, frame_count, fps, x1, y1)

            processor.log_state(tracker.get_active_tracks(), frame_count)
            
            # Write annotated frame to output video
            out.write(frame_bgr)
            
            frame_count += 1
            if frame_count % 10 == 0:
                progress_bar.progress(frame_count / max_frames)

        # Release resources
        cap.release()
        if out: out.release()

        # --- FFmpeg Transcoding (The Web Player Fix) ---
        st.info("Transcoding video to H.264 for web playback...")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", raw_avi_path, 
            "-vcodec", "libx264", "-f", "mp4", final_mp4_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # --- Display Final Output ---
        progress_bar.empty()
        st.success("Analysis Complete!")
        
        st.markdown("### 🎬 Annotated Turnaround Footage")
        st.video(final_mp4_path)
        
        # Clean up the massive RAW avi file
        if os.path.exists(raw_avi_path): os.remove(raw_avi_path)

        st.markdown("---")
        st.markdown("## 📊 Turnaround Operations Report")

        # Process the logged track data to extract KPIs and prepare for visualization
        processor.transform_data(fps=fps)
        df_turnaround = processor.get_dataframe()

        # Top level KPIs
        col1, col2, col3 = st.columns(3)

        kpis = processor.get_kpis()
        total_vehicles = kpis["total_vehicles"]
        total_personnel = kpis["total_personnel"]
        max_duration = kpis["max_duration"]

        col1.metric("🚚 Total Vehicles Detected", total_vehicles)
        col2.metric("👷 Ground Crew Detected", total_personnel)
        col3.metric("⏱️ Longest Service Time", f"{max_duration:.1f}s")


        # Gantt Chart Visualization
        st.markdown("### 📈 Operational Timeline (Gantt)")

        # Plotly Express Timeline creates beautiful, interactive Gantt charts
        fig = px.timeline(
            df_turnaround, 
            x_start=TracksDataColumns.PLOT_START.value, 
            x_end=TracksDataColumns.PLOT_END.value, 
            y=TracksDataColumns.VEHICLE_ID.value,
            color=TracksDataColumns.CATEGORY.value,
            hover_data=[TracksDataColumns.DURATION_SEC.value],
            color_discrete_map={"truck": "#1f77b4", "person": "#ff7f0e"}
        )
        # Update layout to reverse Y axis (so the first vehicle appears at the top)
        fig.update_yaxes(autorange="reversed")
        # Format X axis to just show minutes:seconds
        fig.update_layout(xaxis=dict(tickformat="%M:%S"))

        st.plotly_chart(fig, use_container_width=True)


        # Data export
        st.markdown("### 🗄️ Raw Audit Log")
        # Display the dataframe cleanly
        st.dataframe(
            df_turnaround[[TracksDataColumns.VEHICLE_ID, TracksDataColumns.CATEGORY, TracksDataColumns.START_SEC, TracksDataColumns.END_SEC, TracksDataColumns.DURATION_SEC]], 
            use_container_width=True
        )

        # Provide a download button for the CSV
        csv = df_turnaround.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Turnaround Audit (CSV)",
            data=csv,
            file_name='turnaround_audit_log.csv',
            mime='text/csv',
        )
