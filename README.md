# 🛫 Automated Aircraft Turnaround Tracker

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)

An end-to-end computer vision and operational analytics pipeline designed to measure and optimize aircraft ground turnaround operations. 

This project ingests raw tarmac video streams, utilizes deep learning for object detection, tracks service vehicles across time, and aggregates the data into a post-event analytics dashboard to calculate **Production Efficiency** and **Time-in-Zone** metrics.

## Business Use Case
Ground turnaround time is a major cost center in aviation. This tool transitions raw visual data into structured operational intelligence, allowing operations managers to:
* **Audit Service Level Agreements (SLAs):** Automatically verify how long catering, fuel, and baggage vehicles spend attached to an aircraft.
* **Identify Bottlenecks:** Conduct post-turnaround root cause analysis for delayed departures.

## System Architecture
The pipeline is built from scratch to demonstrate full control over the CV and MLOps lifecycle, deliberately avoiding pre-packaged tracking pipelines in favor of native implementation.

1. **Video Ingestion & Preprocessing:** OpenCV handles frame extraction, spatial cropping (Region of Interest optimization), and color space transformations.
2. **Deep Learning Inference:** A pre-trained `Faster R-CNN (ResNet50)` via Torchvision detects relevant COCO classes (trucks, cars, persons) frame-by-frame.
3. **Temporal Association (Tracking):** A custom-built, stateful **Greedy Intersection over Union (IoU) Tracker** bridges single-frame detections into continuous object lifespans.
4. **Data Engineering:** Object lifespans are logged and transformed into a Pandas DataFrame to extract real-world KPIs (e.g., service duration in minutes).
5. **Insights Dashboard:** A local Streamlit web application serves as the frontend, rendering the annotated video alongside extracted business metrics.

## ⚙️ Installation & Setup

This project utilizes `uv` for reproducible, lockfile-backed dependency management. To clone the repository and install the exact verified environment from the `uv.lock` file:

```bash
git clone [https://github.com/szpwski/aircraft-turnaround-tracker.git](https://github.com/szpwski/aircraft-turnaround-tracker.git)
cd turnaround-tracker
uv sync
```

> Note: For now only exploration notebook has been developed. Rest of work is still WIP.