# 🛫 Automated Aircraft Turnaround Tracker

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

An end-to-end computer vision and operational analytics pipeline designed to measure and optimize aircraft ground turnaround operations. 

This project ingests raw tarmac video files, utilizes deep learning for object detection, tracks service vehicles across time, and aggregates the data into an interactive post-event analytics dashboard to calculate **Production Efficiency** and **Time-in-Zone** metrics.

## [View Live Analytics Dashboard](https://turnaroundtracker-app.redtree-18dff6ae.polandcentral.azurecontainerapps.io)

> **Note:** This application is deployed on a serverless Azure Container App architecture that scales to zero to optimize cloud FinOps. **The initial page load may take 30–45 seconds** as the container cold-starts. Subsequent interactions are instantaneous. 

> **Note 2**: Access requires Microsoft authentication. This is intentionally enabled to demonstrate Azure application security and protected access patterns.

**Want to test the live app?** Since you likely don't have tarmac footage on hand, **[Download this Sample Video (MP4)](https://raw.githubusercontent.com/szpwski/aircraft-turnaround-tracker/main/videos/input_sample/turnaround_video.mp4)** to upload and test the tracking pipeline.

An end-to-end computer vision and operational analytics pipeline.

## Business Use Case
Ground turnaround time is a major cost center in aviation. This tool transitions raw visual data into structured operational intelligence, allowing operations managers to:
* **Audit Service Level Agreements (SLAs):** Automatically verify how long catering, fuel, and baggage vehicles spend attached to an aircraft.
* **Identify Bottlenecks:** Conduct post-turnaround root cause analysis for delayed departures using interactive timeline dashboards.

## System Architecture
The pipeline is built from scratch to demonstrate full control over the CV and MLOps lifecycle, deliberately avoiding pre-packaged tracking pipelines in favor of native implementation.

1. **Video Ingestion:** Users upload raw tarmac video files directly through the Streamlit interface, ensuring a robust, secure data pipeline that avoids third-party streaming dependencies or anti-bot blocks.
2. **Deep Learning Inference:** A pre-trained `Faster R-CNN (ResNet50)` via Torchvision detects relevant COCO classes (trucks, cars, persons). The system utilizes a "Frame Skipper" architecture, running heavy inference at set intervals to optimize CPU compute times.
3. **Temporal Association (Tracking):** A custom-built, stateful **Greedy Intersection over Union (IoU) Tracker** bridges single-frame detections into continuous object lifespans.
4. **Data Engineering:** Object lifespans are logged and transformed into a Pandas DataFrame to extract real-world KPIs.
5. **Insights Dashboard:** A local Streamlit web application orchestrates the pipeline. It handles dynamic OpenCV dimension-locking, uses `FFmpeg` to transcode the processed OpenCV output into a web-safe H.264 format, and renders an interactive `Plotly` Gantt chart of the turnaround timeline.

## Installation & Setup

### Prerequisites
Because the pipeline automatically transcodes the tracked output into a web-safe format, you must have `ffmpeg` installed on your host machine.
* **Mac:** `brew install ffmpeg`
* **Linux:** `sudo apt-get install ffmpeg`
* **Windows:** Install via `winget install ffmpeg`

### Environment Setup
This project utilizes `uv` for reproducible, lockfile-backed dependency management. 

```bash
git clone [https://github.com/szpwski/aircraft-turnaround-tracker.git](https://github.com/szpwski/aircraft-turnaround-tracker.git)
cd aircraft-turnaround-tracker
uv sync
```

### Running the Application
Launch the interactive Streamlit dashboard directly from your terminal:

```bash
uv run streamlit run app.py
```

### Testing & Quality Assurance
This project utilizes pytest for automated unit testing to ensure pipeline reliability and data integrity. Currently, the test suite covers the core utility functions (utils), verifying that bounding box transformations, intersection-over-union (IoU) math, and time-series conversions execute deterministically.

To run the test suite locally using your uv environment:
```bash
uv run pytest
```

### Testing the Pipeline
Once the dashboard opens in your browser, use the sidebar to upload a video file for analysis.

A sample tarmac video has been provided in the repository for testing purposes. Please upload the following file to view the tracking and analytics engine in action:

- `videos/input_sample/turnaround_video.mp4`

(Note: The system contains an automatic 10-second guardrail to prevent massive compute times during local PoC testing).