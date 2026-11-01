# SAR V1.0 — Search and Rescue Aerial Computer Vision Engine

Autonomous search-and-rescue (SAR) aerial computer vision system featuring two-phase fine-tuned YOLO detectors, real-time webcam/UAV stream inference, and audit reporting.

Developed by **Aadarsh Sinha** ([@Monarch666](https://github.com/Monarch666)).

---

## 🚁 Key Capabilities

- **Two-Phase GPU Training Pipeline**: Transfer learning on aerial search-and-rescue datasets (`train_phase1_gpu.py` & `train_phase2_gpu.py`).
- **Live Stream Visualizer Server**: Low-latency video pipeline for drone feed streaming (`visualizer_server.py`).
- **Automated Dataset Auditing**: Class distribution analyzers and report generators (`class_audit_report.txt`, `folder_audit_report.csv`).

---

## 🚀 Quickstart & Setup

```bash
# Clone repository
git clone https://github.com/Monarch666/SAR_V1.0.git
cd SAR_V1.0

# Install dependencies
pip install -r requirements.txt

# Run live inference on webcam
python webcam_inference.py

# Launch visualizer server
python visualizer_server.py
```

---

## 📄 License
MIT License © 2024 [Aadarsh Sinha](https://github.com/Monarch666).
