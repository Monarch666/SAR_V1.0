"""
SAR Drone Detection - Automated Model Evaluator and Reporter
Evaluates the best checkpoint on validation and test splits, and generates a detailed report.
"""

import os
import sys
from ultralytics import YOLO

def generate_report(results_dict, split_name, model_path):
    report_lines = []
    report_lines.append(f"# SAR Drone Detection Model Evaluation - {split_name.upper()} split")
    report_lines.append(f"**Model Path**: `{model_path}`")
    report_lines.append("")
    report_lines.append("## Overall Metrics")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---|")
    
    # Extract keys safely
    mAP50 = results_dict.get("metrics/mAP50(B)", 0.0)
    mAP50_95 = results_dict.get("metrics/mAP50-95(B)", 0.0)
    precision = results_dict.get("metrics/precision(B)", 0.0)
    recall = results_dict.get("metrics/recall(B)", 0.0)
    
    report_lines.append(f"| Precision | {precision:.4f} |")
    report_lines.append(f"| Recall | {recall:.4f} |")
    report_lines.append(f"| mAP50 | {mAP50:.4f} |")
    report_lines.append(f"| mAP50-95 | {mAP50_95:.4f} |")
    report_lines.append("")
    
    report_lines.append("## Class-by-Class Breakdown")
    report_lines.append("| Class ID | Class Name | Precision | Recall | mAP50 | mAP50-95 |")
    report_lines.append("|---|---|---|---|---|---|")
    
    # Class names mapping
    class_names = {
        0: "victim_person",
        1: "rescue_personnel",
        2: "thermal_human",
        3: "fire",
        4: "smoke",
        5: "rescue_vehicle",
        6: "structural_damage"
    }
    
    # YOLO results maps class indexes
    # YOLO v8 results.keys contains classwise maps if available, or we can fetch them from class metrics.
    # Let's extract safely from results object attributes:
    # results.box.p, results.box.r, results.box.map50, results.box.map
    return report_lines

def get_class_metrics(results, class_id):
    try:
        if hasattr(results, 'box') and results.box is not None:
            # results.box.ap_class_index lists the class indices that are present in validation
            if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
                ap_classes = list(results.box.ap_class_index)
                if class_id in ap_classes:
                    idx = ap_classes.index(class_id)
                    p = float(results.box.p[idx]) if hasattr(results.box, 'p') and len(results.box.p) > idx else 0.0
                    r = float(results.box.r[idx]) if hasattr(results.box, 'r') and len(results.box.r) > idx else 0.0
                    ap50 = float(results.box.ap50[idx]) if hasattr(results.box, 'ap50') and len(results.box.ap50) > idx else 0.0
                    ap = float(results.box.ap[idx]) if hasattr(results.box, 'ap') and len(results.box.ap) > idx else 0.0
                    return p, r, ap50, ap
            
            # Fallback to direct indexing only if ap_class_index is not available
            elif hasattr(results.box, 'p') and results.box.p is not None:
                if class_id < len(results.box.p):
                    p = float(results.box.p[class_id])
                    r = float(results.box.r[class_id])
                    ap50 = float(results.box.ap50[class_id])
                    ap = float(results.box.ap[class_id])
                    return p, r, ap50, ap
    except Exception as e:
        print(f"DEBUG: error getting metrics for class {class_id}: {e}")
    return 0.0, 0.0, 0.0, 0.0

def main():
    candidate_paths = [
        r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase2b\weights\best.pt",
        r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase2\weights\best.pt",
        r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase1\weights\best.pt"
    ]
    
    model_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if not model_path:
        print("Error: No trained models found to evaluate.")
        sys.exit(1)
        
    print(f"Loading best available model: {model_path}")
    model = YOLO(model_path)
    
    # Validate on validation split
    print("Evaluating on Validation split...")
    val_results = model.val(data=r"E:\Dataset\merged_final\sar_drone.yaml", split="val", device=0, imgsz=1024, workers=0, batch=8)
    
    # Validate on test split
    print("Evaluating on Test split...")
    test_results = model.val(data=r"E:\Dataset\merged_final\sar_drone.yaml", split="test", device=0, imgsz=1024, workers=0, batch=8)
    
    # Write to a report file
    brain_dir = r"C:\Users\ADARSHASINHA\.gemini\antigravity-ide\brain\29f4fa9e-c17d-47b1-80b6-98fe74e23c74"
    report_file_path = os.path.join(brain_dir, "model_evaluation_report.md")
    
    with open(report_file_path, "w") as f:
        f.write("# Model Evaluation Report\n")
        import datetime
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write validation overall stats
        f.write("## Validation Metrics\n")
        f.write(f"- Precision: {val_results.results_dict.get('metrics/precision(B)', 0.0):.4f}\n")
        f.write(f"- Recall: {val_results.results_dict.get('metrics/recall(B)', 0.0):.4f}\n")
        f.write(f"- mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 0.0):.4f}\n")
        f.write(f"- mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}\n\n")
        
        # Write validation classwise breakdown
        f.write("### Validation Class Breakdown\n")
        f.write("| Class Name | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("|---|---|---|---|---|\n")
        for i, name in val_results.names.items():
            p, r, ap50, ap = get_class_metrics(val_results, i)
            f.write(f"| {name} | {p:.4f} | {r:.4f} | {ap50:.4f} | {ap:.4f} |\n")
            
        f.write("\n---\n\n")
        
        # Write test overall stats
        f.write("## Test Metrics\n")
        f.write(f"- Precision: {test_results.results_dict.get('metrics/precision(B)', 0.0):.4f}\n")
        f.write(f"- Recall: {test_results.results_dict.get('metrics/recall(B)', 0.0):.4f}\n")
        f.write(f"- mAP50: {test_results.results_dict.get('metrics/mAP50(B)', 0.0):.4f}\n")
        f.write(f"- mAP50-95: {test_results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}\n\n")
        
        # Write test classwise breakdown
        f.write("### Test Class Breakdown\n")
        f.write("| Class Name | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("|---|---|---|---|---|\n")
        for i, name in test_results.names.items():
            p, r, ap50, ap = get_class_metrics(test_results, i)
            f.write(f"| {name} | {p:.4f} | {r:.4f} | {ap50:.4f} | {ap:.4f} |\n")
            
    print(f"Report successfully saved to: {report_file_path}")

if __name__ == "__main__":
    main()
