import subprocess
import time
import csv
import os
import torch
from datetime import datetime

def parse_nvidia_smi():
    """
    Parse nvidia-smi output to extract specific metrics
    """
    try:
        # Run nvidia-smi with more detailed parsing
        output = subprocess.check_output(['nvidia-smi', 
            '--query-gpu=timestamp,name,driver_version,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total', 
            '--format=csv,noheader'], 
            universal_newlines=True)
        
        return output.strip()
    except Exception as e:
        print(f"Error collecting GPU metrics: {e}")
        return None

def log_metrics(log_file='gpu_detailed_metrics.csv', interval=60):
    """
    Log GPU metrics periodically
    
    :param log_file: Name of the log file
    :param interval: Logging interval in seconds (default 10 minutes)
    """
    # Ensure log directory exists
    os.makedirs('gpu_logs', exist_ok=True)
    full_path = os.path.join('gpu_logs', log_file)
    
    # Prepare CSV headers
    headers = [
        'timestamp', 'gpu_name', 'driver_version', 
        'temperature', 'power_draw', 'gpu_utilization', 
        'memory_utilization', 'memory_used', 'memory_total'
    ]
    
    # Check if file exists to decide on writing headers
    file_exists = os.path.exists(full_path)
    
    try:
        with open(full_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(headers)
            
            # Continuous logging
            while True:
                # Collect metrics
                metrics_line = parse_nvidia_smi()
                
                if metrics_line:
                    # Split the metrics
                    metrics = metrics_line.split(', ')
                    
                    # Write to CSV
                    writer.writerow(metrics)
                    
                    # Flush to ensure immediate writing
                    csvfile.flush()
                
                # Wait for next interval
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMetrics logging stopped.")
    except Exception as e:
        print(f"Error in logging: {e}")

def simulate_megatron_load():
    """
    Simulate GPU load using PyTorch tensor operations
    """
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available. Cannot simulate GPU load.")
            return False
        
        # Move computation to GPU
        device = torch.device('cuda')
        
        # Create large tensors to stress GPU memory and computation
        print("Generating GPU load...")
        
        # Allocate large tensor
        large_tensor = torch.randn(10000, 10000, device=device)
        
        # Perform computationally intensive operations
        for _ in range(100):
            large_tensor = torch.matmul(large_tensor, large_tensor)
            large_tensor = torch.relu(large_tensor)
        
        print("GPU load simulation complete.")
        return True
    
    except Exception as e:
        print(f"Error simulating GPU load: {e}")
        return False


if __name__ == '__main__':
    # Optional: Simulate some load before logging
    simulate_megatron_load()
    
    # Start logging
    log_metrics()