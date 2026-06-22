"""
SAR Drone Detection - Phase 2b Training (GPU) — OPTIMIZED
Fine-tune ALL layers starting from Phase 1 best weights.
Key changes vs Phase 2:
  - imgsz=1024 (was 640) — captures small SAR targets far better
  - batch=8 (was 16) — to fit 1024 in 16GB VRAM
  - cls=1.5 (was 0.5) — stronger classification signal for imbalanced classes
  - cos_lr=True — cosine annealing avoids LR plateau
  - lr0=0.0005 (was 0.0002) — slightly higher to escape the plateau
Using NVIDIA RTX 2000 Ada (16GB VRAM) with AMP.
"""

import os
from ultralytics import YOLO

def main():
    import os

    last_pt = r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase2b\weights\last.pt"

    print("=" * 60)
    print("Phase 2b: Optimized Fine-Tuning (imgsz=1024, cos_lr)")
    print("=" * 60)

    if os.path.exists(last_pt):
        print(f"Found checkpoint at {last_pt}. Resuming training...")
        model = YOLO(last_pt)
        results = model.train(resume=True)
        print("\n" + "=" * 60)
        print("Phase 2b Training Complete (Resumed)!")
        print("=" * 60)
        print(f"\nBest weights: E:\\Dataset\\runs\\detect\\sar_drone_runs\\yolov8m_sar_v1_phase2b\\weights\\best.pt")
        print(f"Last weights: E:\\Dataset\\runs\\detect\\sar_drone_runs\\yolov8m_sar_v1_phase2b\\weights\\last.pt")
        return

    print("Starting from Phase 1 best.pt weights")
    print("Key optimizations: imgsz=1024, cls=1.5, cos_lr=True")
    print()

    # Load Phase 1 best weights
    model = YOLO(r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase1\weights\best.pt")

    # Phase 2b: Unfreeze everything, higher resolution, stronger cls loss
    results = model.train(
        data=r"E:\Dataset\merged_final\sar_drone.yaml",
        epochs=50,                    # More epochs — cosine LR needs room
        batch=8,                      # Halved from 16 to fit imgsz=1024 in 16GB
        imgsz=1024,                   # *** KEY: Higher resolution for small targets ***
        device=0,                     # GPU 0
        amp=True,                     # Mixed precision
        workers=0,                    # Safe main-process dataloading on Windows to avoid MemoryErrors
        project=r"E:\Dataset\runs\detect\sar_drone_runs",
        name="yolov8m_sar_v1_phase2b",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,                   # Higher than Phase 2 (was 0.0002)
        lrf=0.01,                     # Final LR = lr0 * lrf
        cos_lr=True,                  # *** KEY: Cosine annealing schedule ***
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        freeze=0,                     # Unfreeze all layers
        patience=20,                  # Early stopping — generous with cosine LR
        save_period=5,                # Save checkpoint every 5 epochs
        seed=0,
        deterministic=True,
        # Loss weights — boosted cls for class imbalance
        box=7.5,
        cls=1.5,                      # *** KEY: 3x higher than before (was 0.5) ***
        dfl=1.5,
        # Augmentation — moderate for fine-tuning at higher resolution
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.2,
        erasing=0.3,
        close_mosaic=10,
        auto_augment="randaugment",
        # Output
        plots=True,
        verbose=True,
        save=True,
    )

    print("\n" + "=" * 60)
    print("Phase 2b Training Complete!")
    print("=" * 60)

    # Print key metrics
    if results and hasattr(results, 'results_dict'):
        rd = results.results_dict
        print(f"  mAP50:     {rd.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95:  {rd.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Precision: {rd.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall:    {rd.get('metrics/recall(B)', 'N/A')}")

    print(f"\nBest weights: E:\\Dataset\\runs\\detect\\sar_drone_runs\\yolov8m_sar_v1_phase2b\\weights\\best.pt")
    print(f"Last weights: E:\\Dataset\\runs\\detect\\sar_drone_runs\\yolov8m_sar_v1_phase2b\\weights\\last.pt")


if __name__ == "__main__":
    main()
