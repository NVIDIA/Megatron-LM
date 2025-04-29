import subprocess
import time
import csv
import os
import torch
from datetime import datetime
import threading

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

def log_metrics(log_file='gpu_detailed_metrics_intenseload.csv', interval=60):
    """
    Log GPU metrics periodically
    
    :param log_file: Name of the log file
    :param interval: Logging interval in seconds
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
            iteration = 1
            while True:
                print(f"\n--- Stress Test Iteration {iteration} ---")
                
                # Run GPU stress test
                start_stress_time = time.time()
                intensive_gpu_load()
                end_stress_time = time.time()
                print(f"Stress Test Duration: {end_stress_time - start_stress_time:.2f} seconds")
                
                # Collect and log metrics
                metrics_line = parse_nvidia_smi()
                
                if metrics_line:
                    # Split the metrics
                    metrics = metrics_line.split(', ')
                    
                    # Write to CSV
                    writer.writerow(metrics)
                    
                    # Flush to ensure immediate writing
                    csvfile.flush()
                
                # Increment iteration
                iteration += 1
                
                # Wait for next interval
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMetrics logging stopped.")
    except Exception as e:
        print(f"Error in logging: {e}")

def intensive_gpu_load():
    """
    Create multiple intensive GPU load scenarios with increasing complexity
    """
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA not available!")
            return False
        
        # Select GPU device
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Multiple load generation techniques with progressive complexity
        def matrix_multiplication_stress(complexity=1):
            print(f"Running Matrix Multiplication Stress Test (Complexity: {complexity})...")
            # Increase tensor sizes and multiplication iterations with complexity
            sizes = [5000 * complexity, 10000 * complexity, 15000 * complexity]
            for size in sizes:
                print(f"Creating {size}x{size} tensor operations")
                large_tensor = torch.randn(size, size, device=device)
                
                # Increase iterations with complexity
                for _ in range(50 * complexity):
                    large_tensor = torch.matmul(large_tensor, large_tensor)
                    large_tensor = torch.relu(large_tensor)
                
                # Force garbage collection
                del large_tensor
                torch.cuda.empty_cache()
        
        def parallel_tensor_operations(complexity=1):
            print(f"Running Parallel Tensor Operations (Complexity: {complexity})...")
            # Increase number of streams and operations with complexity
            streams = [torch.cuda.Stream() for _ in range(4 * complexity)]
            
            def stream_task(stream):
                with torch.cuda.stream(stream):
                    tensor = torch.randn(5000 * complexity, 5000 * complexity, device=device)
                    for _ in range(30 * complexity):
                        tensor = torch.sin(tensor)
                        tensor = torch.exp(tensor)
            
            # Run tasks on different streams
            for stream in streams:
                stream_task(stream)
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
        
        def gpu_memory_stress(complexity=1):
            print(f"Running GPU Memory Stress Test (Complexity: {complexity})...")
            # Increase number and size of tensors with complexity
            tensors = [
                torch.randn(10000 * complexity, 10000 * complexity, device=device) 
                for _ in range(10 * complexity)
            ]
            
            # Perform operations to prevent optimization
            for tensor in tensors:
                torch.max(tensor)
            
            # Clear memory
            del tensors
            torch.cuda.empty_cache()
        
        # Execute stress tests with increasing complexity
        matrix_multiplication_stress(1)
        parallel_tensor_operations(1)
        gpu_memory_stress(1)
        
        print("Comprehensive GPU Load Simulation Complete!")
        return True
    
    except Exception as e:
        print(f"GPU Load Simulation Error: {e}")
        return False

if __name__ == '__main__':
    # Start logging metrics and running stress tests
    log_metrics()