import subprocess
import time
import csv
import os
from datetime import datetime

def get_gpu_metrics():
    try:
        # Run nvidia-smi command to get metrics
        output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        
        # Parse key metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'raw_output': output
        }
        
        return metrics
    except Exception as e:
        print(f"Error collecting GPU metrics: {e}")
        return None

def log_metrics(log_file='gpu_metrics.csv'):
    # Ensure log directory exists
    os.makedirs('gpu_logs', exist_ok=True)
    full_path = os.path.join('gpu_logs', log_file)
    
    # Check if file exists to decide on writing headers
    file_exists = os.path.exists(full_path)
    
    with open(full_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'raw_output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        while True:
            metrics = get_gpu_metrics()
            if metrics:
                writer.writerow(metrics)
            
            # Log every 10 minutes as per your project plan
            time.sleep(600)

if __name__ == '__main__':
    log_metrics()