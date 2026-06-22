"""
Live Training Visualizer Server for YOLOv8
Provides a premium web UI to monitor training loss, metrics, batch images, and live logs.
Run this script and open http://localhost:8000 in your browser.
"""

import os
import json
import re
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8000

def get_run_dir():
    base = r"E:\Dataset\runs\detect\sar_drone_runs"
    if not os.path.exists(base):
        return r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase1"
    subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        return r"E:\Dataset\runs\detect\sar_drone_runs\yolov8m_sar_v1_phase1"
    return max(subdirs, key=os.path.getmtime)

def get_log_path():
    base_brain = r"C:\Users\ADARSHASINHA\.gemini\antigravity-ide\brain"
    if os.path.exists(base_brain):
        subdirs = [os.path.join(base_brain, d) for d in os.listdir(base_brain) if os.path.isdir(os.path.join(base_brain, d))]
        if subdirs:
            # Sort subdirectories so the most recently modified is checked first
            subdirs.sort(key=os.path.getmtime, reverse=True)
            for subdir in subdirs:
                tasks_dir = os.path.join(subdir, ".system_generated", "tasks")
                if os.path.exists(tasks_dir):
                    log_files = [os.path.join(tasks_dir, f) for f in os.listdir(tasks_dir) if f.endswith(".log")]
                    if log_files:
                        return max(log_files, key=os.path.getmtime)
    
    fallback_dir = r"C:\Users\ADARSHASINHA\.gemini\antigravity-ide\brain\29f4fa9e-c17d-47b1-80b6-98fe74e23c74\.system_generated\tasks"
    if os.path.exists(fallback_dir):
        log_files = [os.path.join(fallback_dir, f) for f in os.listdir(fallback_dir) if f.endswith(".log")]
        if log_files:
            return max(log_files, key=os.path.getmtime)
            
    return r"C:\Users\ADARSHASINHA\.gemini\antigravity-ide\brain\26418ec8-3977-4f71-ae8b-90addbbd8af0\.system_generated\tasks\task-43.log"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 SAR Drone Training Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --card-border: #1f2937;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #38bdf8;
            --accent-secondary: #a855f7;
            --accent-success: #34d399;
            --accent-warning: #fbbf24;
            --terminal-bg: #030712;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
            overflow-x: hidden;
        }

        /* Ambient Glow Backgrounds */
        .glow {
            position: absolute;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, rgba(0,0,0,0) 70%);
            top: -100px;
            right: -100px;
            z-index: -1;
            pointer-events: none;
        }
        .glow-2 {
            position: absolute;
            width: 500px;
            height: 500px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.05) 0%, rgba(0,0,0,0) 70%);
            bottom: -150px;
            left: -150px;
            z-index: -1;
            pointer-events: none;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
        }

        .logo-container h1 {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .logo-container p {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(52, 211, 153, 0.1);
            border: 1px solid rgba(52, 211, 153, 0.2);
            color: var(--accent-success);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-success);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.5; }
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }

        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            border-color: rgba(56, 189, 248, 0.3);
            transform: translateY(-2px);
        }

        .stat-label {
            font-size: 13px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Tabs Navigation */
        .tabs {
            display: flex;
            gap: 12px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 8px;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 16px;
            font-weight: 500;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }

        .tab-btn.active {
            color: var(--accent-primary);
            background: rgba(56, 189, 248, 0.1);
        }

        /* Tab Contents */
        .tab-content {
            display: none;
            animation: fadeIn 0.4s ease;
        }

        .tab-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Charts Layout */
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
            gap: 24px;
        }

        @media (max-width: 768px) {
            .charts-container {
                grid-template-columns: 1fr;
            }
        }

        .chart-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            height: 380px;
        }

        .chart-card h3 {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }

        .chart-wrapper {
            position: relative;
            height: calc(100% - 32px);
            width: 100%;
        }

        /* Live Logs Panel */
        .logs-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .logs-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .terminal {
            background-color: var(--terminal-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            line-height: 1.6;
            height: 480px;
            overflow-y: auto;
            color: #d1d5db;
            white-space: pre-wrap;
        }

        /* Batch Images */
        .batches-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 24px;
        }

        .batch-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .batch-card:hover {
            border-color: var(--accent-primary);
            transform: scale(1.01);
        }

        .batch-card img {
            width: 100%;
            height: auto;
            display: block;
            object-fit: cover;
        }

        .batch-info {
            padding: 16px;
            border-top: 1px solid var(--card-border);
        }

        .batch-title {
            font-size: 15px;
            font-weight: 600;
        }

        .batch-desc {
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="glow"></div>
    <div class="glow-2"></div>
    
    <div class="container">
        <!-- Header -->
        <header>
            <div class="logo-container">
                <h1>YOLOv8 SAR Drone Training</h1>
                <p>NVIDIA RTX 2000 Ada • 16GB VRAM • PyTorch + CUDA 13.0</p>
            </div>
            <div class="status-badge" id="status-badge">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">INITIALIZING</span>
            </div>
        </header>

        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-label">Current Epoch</span>
                <span class="stat-value" id="stat-epoch">-</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Train Box Loss</span>
                <span class="stat-value" id="stat-loss-box">-</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Train Cls Loss</span>
                <span class="stat-value" id="stat-loss-cls">-</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">mAP50</span>
                <span class="stat-value" id="stat-map50" style="color: var(--accent-success);">-</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">mAP50-95</span>
                <span class="stat-value" id="stat-map50-95" style="color: var(--accent-primary);">-</span>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('metrics')">Training Curves</button>
            <button class="tab-btn" onclick="switchTab('batches')">Batch Samples</button>
            <button class="tab-btn" onclick="switchTab('logs')">Live Logs</button>
        </div>

        <!-- Tab 1: Curves -->
        <div id="tab-metrics" class="tab-content active">
            <div class="charts-container">
                <div class="chart-card">
                    <h3>Box Loss (Bounding Box Regression)</h3>
                    <div class="chart-wrapper">
                        <canvas id="boxLossChart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <h3>Classification Loss</h3>
                    <div class="chart-wrapper">
                        <canvas id="clsLossChart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <h3>mAP50 & mAP50-95</h3>
                    <div class="chart-wrapper">
                        <canvas id="mapChart"></canvas>
                    </div>
                </div>
                <div class="chart-card">
                    <h3>DFL Loss (Distribution Focal Loss)</h3>
                    <div class="chart-wrapper">
                        <canvas id="dflLossChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Batches -->
        <div id="tab-batches" class="tab-content">
            <div class="batches-grid">
                <div class="batch-card">
                    <img id="img-batch0" src="" alt="Waiting for train_batch0.jpg..." onerror="this.src='https://placehold.co/600x400/111827/9ca3af?text=Waiting+for+train_batch0.jpg'">
                    <div class="batch-info">
                        <div class="batch-title">train_batch0.jpg</div>
                        <div class="batch-desc">First batch of augmented training labels overlaid on training images.</div>
                    </div>
                </div>
                <div class="batch-card">
                    <img id="img-batch1" src="" alt="Waiting for train_batch1.jpg..." onerror="this.src='https://placehold.co/600x400/111827/9ca3af?text=Waiting+for+train_batch1.jpg'">
                    <div class="batch-info">
                        <div class="batch-title">train_batch1.jpg</div>
                        <div class="batch-desc">Second batch of augmented training labels showing data diversity.</div>
                    </div>
                </div>
                <div class="batch-card">
                    <img id="img-batch2" src="" alt="Waiting for train_batch2.jpg..." onerror="this.src='https://placehold.co/600x400/111827/9ca3af?text=Waiting+for+train_batch2.jpg'">
                    <div class="batch-info">
                        <div class="batch-title">train_batch2.jpg</div>
                        <div class="batch-desc">Third batch of augmented training labels demonstrating scaling and flipping.</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 3: Logs -->
        <div id="tab-logs" class="tab-content">
            <div class="logs-card">
                <div class="logs-header">
                    <h3>Real-time training terminal output</h3>
                    <span id="log-time" style="font-size: 13px; color: var(--text-secondary);">Last updated: -</span>
                </div>
                <div class="terminal" id="terminal-content">Loading logs...</div>
            </div>
        </div>
    </div>

    <script>
        let boxChart, clsChart, mapChart, dflChart;

        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(31, 41, 55, 0.5)' },
                        title: { display: true, text: 'Epoch', color: '#9ca3af' },
                        ticks: { color: '#9ca3af' }
                    },
                    y: {
                        grid: { color: 'rgba(31, 41, 55, 0.5)' },
                        ticks: { color: '#9ca3af' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f3f4f6' } }
                }
            };

            boxChart = new Chart(document.getElementById('boxLossChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Train Box Loss', data: [], borderColor: '#38bdf8', tension: 0.1, borderWidth: 2 },
                        { label: 'Val Box Loss', data: [], borderColor: '#fbbf24', tension: 0.1, borderWidth: 2 }
                    ]
                },
                options: chartOptions
            });

            clsChart = new Chart(document.getElementById('clsLossChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Train Cls Loss', data: [], borderColor: '#a855f7', tension: 0.1, borderWidth: 2 },
                        { label: 'Val Cls Loss', data: [], borderColor: '#f43f5e', tension: 0.1, borderWidth: 2 }
                    ]
                },
                options: chartOptions
            });

            mapChart = new Chart(document.getElementById('mapChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'mAP50', data: [], borderColor: '#34d399', tension: 0.1, borderWidth: 2 },
                        { label: 'mAP50-95', data: [], borderColor: '#38bdf8', tension: 0.1, borderWidth: 2 }
                    ]
                },
                options: chartOptions
            });

            dflChart = new Chart(document.getElementById('dflLossChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Train DFL Loss', data: [], borderColor: '#22c55e', tension: 0.1, borderWidth: 2 },
                        { label: 'Val DFL Loss', data: [], borderColor: '#eab308', tension: 0.1, borderWidth: 2 }
                    ]
                },
                options: chartOptions
            });
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }

        async function fetchData() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();

                // Update Status
                const statusBadge = document.getElementById('status-badge');
                const statusDot = document.getElementById('status-dot');
                const statusText = document.getElementById('status-text');
                
                if (data.status === 'RUNNING') {
                    statusBadge.style.color = 'var(--accent-success)';
                    statusBadge.style.borderColor = 'rgba(52, 211, 153, 0.2)';
                    statusBadge.style.background = 'rgba(52, 211, 153, 0.1)';
                    statusDot.style.backgroundColor = 'var(--accent-success)';
                    statusDot.style.boxShadow = '0 0 8px var(--accent-success)';
                    statusText.textContent = 'TRAINING RUNNING';
                } else {
                    statusBadge.style.color = 'var(--text-secondary)';
                    statusBadge.style.borderColor = 'var(--card-border)';
                    statusBadge.style.background = 'rgba(255, 255, 255, 0.05)';
                    statusDot.style.backgroundColor = 'var(--text-secondary)';
                    statusDot.style.boxShadow = 'none';
                    statusText.textContent = data.status;
                }

                // Update Logs
                const terminal = document.getElementById('terminal-content');
                const atBottom = terminal.scrollHeight - terminal.clientHeight <= terminal.scrollTop + 50;
                terminal.textContent = data.logs;
                if (atBottom) {
                    terminal.scrollTop = terminal.scrollHeight;
                }
                document.getElementById('log-time').textContent = 'Last updated: ' + new Date().toLocaleTimeString();

                // Update Image sources with cache bust
                const rand = Math.random();
                document.getElementById('img-batch0').src = '/runs/train_batch0.jpg?cb=' + rand;
                document.getElementById('img-batch1').src = '/runs/train_batch1.jpg?cb=' + rand;
                document.getElementById('img-batch2').src = '/runs/train_batch2.jpg?cb=' + rand;

                // Update Stats and Charts if we have metrics
                if (data.epochs && data.epochs.length > 0) {
                    const lastIdx = data.epochs.length - 1;
                    const totalEpochs = data.total_epochs || 20;
                    document.getElementById('stat-epoch').textContent = data.epochs[lastIdx] + ' / ' + totalEpochs;
                    document.getElementById('stat-loss-box').textContent = Number(data.train_box_loss[lastIdx]).toFixed(4);
                    document.getElementById('stat-loss-cls').textContent = Number(data.train_cls_loss[lastIdx]).toFixed(4);
                    document.getElementById('stat-map50').textContent = Number(data.metrics_map50[lastIdx]).toFixed(4);
                    document.getElementById('stat-map50-95').textContent = Number(data.metrics_map50_95[lastIdx]).toFixed(4);

                    // Update charts data
                    boxChart.data.labels = data.epochs;
                    boxChart.data.datasets[0].data = data.train_box_loss;
                    boxChart.data.datasets[1].data = data.val_box_loss;
                    boxChart.update();

                    clsChart.data.labels = data.epochs;
                    clsChart.data.datasets[0].data = data.train_cls_loss;
                    clsChart.data.datasets[1].data = data.val_cls_loss;
                    clsChart.update();

                    mapChart.data.labels = data.epochs;
                    mapChart.data.datasets[0].data = data.metrics_map50;
                    mapChart.data.datasets[1].data = data.metrics_map50_95;
                    mapChart.update();

                    dflChart.data.labels = data.epochs;
                    dflChart.data.datasets[0].data = data.train_dfl_loss;
                    dflChart.data.datasets[1].data = data.val_dfl_loss;
                    dflChart.update();
                } else {
                    // Try parsing current epoch and total epochs from logs
                    const epochMatch = [...data.logs.matchAll(/Epoch\\s+(\\d+)\\/(\\d+)/g)];
                    if (epochMatch.length > 0) {
                        const currentEpoch = epochMatch[epochMatch.length - 1][1];
                        const totalEpochs = epochMatch[epochMatch.length - 1][2];
                        document.getElementById('stat-epoch').textContent = currentEpoch + ' / ' + totalEpochs;
                    }
                }
            } catch (err) {
                console.error("Error fetching metrics:", err);
            }
        }

        window.onload = () => {
            initCharts();
            fetchData();
            setInterval(fetchData, 4000);
        };
    </script>
</body>
</html>
"""


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Read logs
            logs = "No log data available."
            status = "UNKNOWN"
            log_path = get_log_path()
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        # Get last 150 lines to keep it readable but detailed
                        logs = "".join(lines[-150:])
                        
                        # Determine run status
                        full_content = "".join(lines)
                        if "Phase 2b Training Complete!" in full_content:
                            status = "COMPLETE (PHASE 2b)"
                        elif "Phase 2 Fine-Tuning Complete!" in full_content:
                            status = "COMPLETE (PHASE 2)"
                        elif "Phase 1 Training Complete!" in full_content:
                            status = "COMPLETE (PHASE 1)"
                        elif "Traceback" in full_content or "Error" in full_content:
                            status = "ERROR"
                        elif any("Scanning" in l or "Epoch" in l or "val:" in l for l in lines[-10:]):
                            status = "RUNNING"
                        else:
                            status = "RUNNING"
                except Exception as e:
                    logs = f"Error reading log file: {str(e)}"

            # Read CSV metrics
            csv_path = os.path.join(get_run_dir(), "results.csv")
            epochs = []
            train_box_loss = []
            train_cls_loss = []
            train_dfl_loss = []
            val_box_loss = []
            val_cls_loss = []
            val_dfl_loss = []
            metrics_map50 = []
            metrics_map50_95 = []

            if os.path.exists(csv_path):
                try:
                    with open(csv_path, "r") as f:
                        lines = f.read().strip().split("\n")
                    if len(lines) > 1:
                        headers = [h.strip() for h in lines[0].split(",")]
                        # Mapping headers to columns
                        for line in lines[1:]:
                            cols = [c.strip() for c in line.split(",")]
                            if len(cols) == len(headers):
                                # Convert row to a dict
                                row = dict(zip(headers, cols))
                                # Key mapping (YOLOv8 keys can have spaces or varying names)
                                epoch = int(float(row.get("epoch", 0)))
                                epochs.append(epoch)
                                
                                train_box_loss.append(float(row.get("train/box_loss", 0)))
                                train_cls_loss.append(float(row.get("train/cls_loss", 0)))
                                train_dfl_loss.append(float(row.get("train/dfl_loss", 0)))
                                
                                val_box_loss.append(float(row.get("val/box_loss", 0)))
                                val_cls_loss.append(float(row.get("val/cls_loss", 0)))
                                val_dfl_loss.append(float(row.get("val/dfl_loss", 0)))
                                
                                metrics_map50.append(float(row.get("metrics/mAP50(B)", 0)))
                                metrics_map50_95.append(float(row.get("metrics/mAP50-95(B)", 0)))
                except Exception as e:
                    print(f"Error parsing results.csv: {e}")

            # Read total_epochs from args.yaml
            total_epochs = 20  # default fallback
            args_path = os.path.join(get_run_dir(), "args.yaml")
            if os.path.exists(args_path):
                try:
                    with open(args_path, "r") as f:
                        for line in f:
                            if line.strip().startswith("epochs:"):
                                total_epochs = int(line.strip().split(":")[1].strip())
                                break
                except Exception as e:
                    print(f"Error parsing args.yaml: {e}")

            response_data = {
                "status": status,
                "logs": logs,
                "epochs": epochs,
                "total_epochs": total_epochs,
                "train_box_loss": train_box_loss,
                "train_cls_loss": train_cls_loss,
                "train_dfl_loss": train_dfl_loss,
                "val_box_loss": val_box_loss,
                "val_cls_loss": val_cls_loss,
                "val_dfl_loss": val_dfl_loss,
                "metrics_map50": metrics_map50,
                "metrics_map50_95": metrics_map50_95,
            }
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        elif path.startswith("/runs/"):
            filename = os.path.basename(path)
            # Prevent directory traversal
            clean_filename = re.sub(r"[^a-zA-Z0-9_\-\.]", "", filename)
            filepath = os.path.join(get_run_dir(), clean_filename)


            if os.path.exists(filepath):
                self.send_response(200)
                if filepath.endswith(".jpg") or filepath.endswith(".jpeg"):
                    self.send_header("Content-Type", "image/jpeg")
                elif filepath.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                else:
                    self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                try:
                    with open(filepath, "rb") as f:
                        self.wfile.write(f.read())
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        self.send_response(404)
        self.end_headers()


def run():
    print(f"Starting server on http://localhost:{PORT}")
    server = HTTPServer(("0.0.0.0", PORT), VisualizerRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == "__main__":
    run()
