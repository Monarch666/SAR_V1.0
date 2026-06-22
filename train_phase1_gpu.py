"""
SAR Drone Detection - Phase 1 Training (GPU)
YOLOv8m with frozen backbone (first 10 layers) for transfer learning.
Using NVIDIA RTX 2000 Ada (16GB VRAM) with AMP for faster training.
"""

from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8m
    model = YOLO("yolov8m.pt")

    # Phase 1: Train with frozen backbone (transfer learning)
    # Same config as previous CPU attempt, but now on GPU with AMP
    results = model.train(
        data=r"E:\Dataset\merged_final\sar_drone.yaml",
        epochs=20,
        batch=16,
        imgsz=640,
        device=0,                    # GPU 0
        amp=True,                    # Mixed precision for faster training
        workers=4,
        project=r"E:\Dataset\runs\detect\sar_drone_runs",
        name="yolov8m_sar_v1_phase1",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        freeze=10,                   # Freeze first 10 layers (backbone)
        patience=30,
        save_period=10,
        seed=0,
        deterministic=True,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.3,
        erasing=0.4,
        close_mosaic=10,
        auto_augment="randaugment",
        # Output
        plots=True,
        verbose=True,
        save=True,
    )

    print("\n" + "=" * 60)
    print("Phase 1 Training Complete!")
    print("=" * 60)

    # Print key metrics
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        print(f"  mAP50:     {rd.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95:  {rd.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Precision: {rd.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall:    {rd.get('metrics/recall(B)', 'N/A')}")

    print(f"\nBest weights: E:\\Dataset\\runs\\detect\\sar_drone_runs\\yolov8m_sar_v1_phase1\\weights\\best.pt")
    print(f"Last weights: E:\\Dataset\\runs\\detect\\sar_drone_runs\\yolov8m_sar_v1_phase1\\weights\\last.pt")


if __name__ == "__main__":
    main()
