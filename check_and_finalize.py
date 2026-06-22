"""
SAR Drone Detection - Monitor and Finalize Script (Robust version with psutil)
Monitors training progress, and if it is Monday morning (or training is complete),
stops training, evaluates the model, exports it to ONNX, and generates a final report.
"""

import os
import sys
import time
import datetime
import subprocess
import re
import psutil
from ultralytics import YOLO

def is_training_running():
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and any('train_phase2_gpu.py' in part for part in cmd):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def check_training_status():
    base_brain = r"C:\Users\ADARSHASINHA\.gemini\antigravity-ide\brain\29f4fa9e-c17d-47b1-80b6-98fe74e23c74\.system_generated\tasks"
    if not os.path.exists(base_brain):
        return "Unknown", 0, ""
        
    log_files = [os.path.join(base_brain, f) for f in os.listdir(base_brain) if f.endswith(".log")]
    if not log_files:
        return "Unknown", 0, ""
        
    # Filter for log files that contain our training indicators
    training_logs = []
    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "Phase 2b: Optimized Fine-Tuning" in content or "Starting training for 50 epochs..." in content:
                    training_logs.append((log_file, content))
        except Exception:
            pass
            
    if not training_logs:
        return "Stopped", 0, ""
        
    # Get the latest training log file
    latest_log_file, content = max(training_logs, key=lambda x: os.path.getmtime(x[0]))
    
    # Check training progress in this log
    current_epoch = 0
    
    # Find lines with epoch progression like " 15/50 "
    epochs_found = [int(x) for x in re.findall(r"\s+(\d+)/50\s+", content)]
    if epochs_found:
        current_epoch = max(epochs_found)
        
    is_running = is_training_running()
    return "Running" if is_running else "Stopped", current_epoch, content

def main():
    print(f"[{datetime.datetime.now()}] Running monitor check...")
    
    # Determine current day
    now = datetime.datetime.now()
    # Monday is weekday 0, Tuesday is 1, etc.
    is_monday_or_later = now.weekday() == 0 or now.weekday() > 0 and now.weekday() < 5
    # Let's check if the hour is 7 AM or later on Monday
    time_to_finalize = is_monday_or_later and now.hour >= 7
    
    # For testing, we can force finalization if training is complete
    best_weights = r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase2b\weights\best.pt"
    training_complete = False
    
    status, current_epoch, log_content = check_training_status()
    print(f"Training status: {status}, Current Epoch: {current_epoch}/50")
    
    if "Phase 2b Training Complete!" in log_content or "Phase 2b Training Complete (Resumed)!" in log_content:
        training_complete = True

    if time_to_finalize or training_complete:
        print("Finalization condition met. Proceeding to finalize model...")
        
        # 1. Kill the training python task if still running
        try:
            killed = False
            for proc in psutil.process_iter(['cmdline']):
                cmd = proc.info['cmdline']
                if cmd and any('train_phase2_gpu.py' in part for part in cmd):
                    print(f"Terminating training process {proc.pid}...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        print(f"Process {proc.pid} did not exit. Killing it...")
                        proc.kill()
                    killed = True
            if not killed:
                print("No active training process found to terminate.")
        except Exception as e:
            print(f"Error terminating process: {e}")
                
        # Wait for files to unlock
        time.sleep(5)
        
        # 2. Run the evaluator and reporter
        eval_script = r"E:\Dataset\evaluate_and_report.py"
        if os.path.exists(eval_script):
            print("Running evaluate_and_report.py...")
            subprocess.run([sys.executable, eval_script])
            
        # 3. Export model to ONNX format for deployment
        if os.path.exists(best_weights):
            print("Exporting best model to ONNX format...")
            try:
                model = YOLO(best_weights)
                model.export(format="onnx", imgsz=1024, dynamic=True)
                print("Model successfully exported to ONNX format!")
            except Exception as e:
                print(f"Error exporting model: {e}")
        else:
            print(f"Error: Best weights not found at {best_weights}")
            
        print("Finalization complete!")
    else:
        if status == "Stopped" and not training_complete:
            print("WARNING: Training is stopped, but training is not complete and it's not Monday yet.")
            print("RESTART_TRAINING")
        else:
            print("Finalization condition not met yet. Training will continue.")

if __name__ == "__main__":
    main()
