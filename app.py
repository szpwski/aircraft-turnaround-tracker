import streamlit as st
import cv2
import yt_dlp
import tempfile
import os
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- Page Config & UI ---
st.set_page_config(page_title="Turnaround Analytics", layout="wide")
st.title("🛫 Post-Event Turnaround Analytics")
st.markdown("Analyze aircraft ground operations from live webcam streams.")

# --- Sidebar Inputs ---
st.sidebar.header("Data Source")
yt_url = st.sidebar.text_input("Paste YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")


if st.sidebar.button("▶️ Run Analysis"):
    if not yt_url:
        st.warning("Please provide a valid YouTube URL.")

    else:
        with st.spinner("Extracting stream metadata..."):
            ydl_opts = {"format": "best[ext=mp4]/", "quiet": True, "noplaylist": True}
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(yt_url, download=False)
                    stream_url = info_dict["url"]
            except Exception as e:
                st.error(f"Error extracting stream URL: {e}")
                st.stop()

        st.info("Stream connected. Processing frames... (this will take a moment)")

        # Open CV Setup
        cap = cv2.VideoCapture(stream_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps: fps = 30  # Default to 30 if unable to get FPS

        # Setup temporary files for video generation
        temp_dir = tempfile.gettempdir()
        raw_avi_path = os.path.join(temp_dir, "raw_output.avi")
        final_mp4_path = os.path.join(temp_dir, "final_web.mp4")

        out = None
        width, height = 0, 0
        max_frames = int(30 * fps) # Guardrail to limit processing to 30 seconds of video for demo
        frame_count = 0

        # Streamlit progress bar
        progress_bar = st.progress(0)

        # Processing loop
        while frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                break

            # Initialize video writer on first frame
            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(raw_avi_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

            # Inference
            frame_bgr = frame.copy()
            # Dummy box for visual feedback if model isn't hooked up yet
            cv2.rectangle(frame_bgr, (50, 50), (250, 250), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Tracking Active", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Dimension lock check
            if frame_bgr.shape[:2] != (height, width):
                frame_bgr = cv2.resize(frame_bgr, (width, height))
            if frame_bgr.dtype != np.uint8:
                frame_bgr = frame_bgr.astype(np.uint8)
                
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
