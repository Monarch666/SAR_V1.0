import sys
from ultralytics import YOLO

def main():
    weights_path = r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase2b\weights\best.pt"
    print("=" * 60)
    print("SAR Drone Detection - Live Webcam Inference")
    print("=" * 60)
    print(f"Loading model from: {weights_path}")
    
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    print("\nStarting webcam feed (source=0)...")
    print("Press 'q' inside the video window to quit.")
    
    try:
        # Run live inference on default webcam (source=0)
        # conf=0.25 keeps a good balance of precision vs recall
        model.predict(source=0, show=True, imgsz=1024, conf=0.25)
    except Exception as e:
        print(f"\nError running live inference: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the webcam is plugged in and recognized by Windows.")
        print("2. Make sure no other app (e.g. Teams, Zoom, Camera app) is currently using the webcam.")
        print("3. If you have multiple webcams/video capture cards, try editing the script to use source=1, source=2, etc.")

if __name__ == "__main__":
    main()
