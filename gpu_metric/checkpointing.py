import os
import json
import time
import torch
import csv
import numpy as np
import subprocess
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpu_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gpu_monitor")

class GPUMetricsCollector:
    """Collects and analyzes GPU metrics for failure prediction"""
    
    def __init__(
        self, 
        log_file: str = 'gpu_metrics.csv',
        interval: int = 10,  # seconds
        alert_thresholds: Dict = None
    ):
        self.log_file = os.path.join('gpu_logs', log_file)
        self.interval = interval
        self.running = False
        self.metrics_history = {}  # Store recent metrics by GPU ID
        self.history_window = 60  # Number of data points to keep per GPU
        
        # Default thresholds - adjust based on your specific GPU model and environment
        self.alert_thresholds = alert_thresholds or {
            'temperature': 85,  # Celsius
            'memory_utilization': 95,  # Percent
            'power_fluctuation': 15,  # Percent change between readings
            'errors': 0,  # Any errors are concerning
            'gpu_utilization_drop': 30,  # Percent drop from baseline
            'temperature_rise_rate': 5,  # Celsius per minute
        }
        
        # Ensure log directory exists
        os.makedirs('gpu_logs', exist_ok=True)
        
        # Initialize CSV headers if file doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'gpu_id', 'gpu_name', 'driver_version', 
                    'temperature', 'power_draw', 'gpu_utilization', 
                    'memory_utilization', 'memory_used', 'memory_total',
                    'errors', 'failure_risk'
                ])
    
    def start_collection(self):
        """Start collecting GPU metrics in a background thread"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Started GPU metrics collection")
    
    def stop_collection(self):
        """Stop the metrics collection thread"""
        self.running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
        logger.info("Stopped GPU metrics collection")
    
    def _collection_loop(self):
        """Main loop for collecting metrics"""
        while self.running:
            try:
                # Collect metrics from all GPUs
                metrics_list = self._collect_metrics()
                
                # Process and analyze metrics
                for gpu_metrics in metrics_list:
                    gpu_id = gpu_metrics['gpu_id']
                    
                    # Update history for this GPU
                    if gpu_id not in self.metrics_history:
                        self.metrics_history[gpu_id] = []
                    
                    self.metrics_history[gpu_id].append(gpu_metrics)
                    
                    # Trim history to window size
                    if len(self.metrics_history[gpu_id]) > self.history_window:
                        self.metrics_history[gpu_id] = self.metrics_history[gpu_id][-self.history_window:]
                    
                    # Calculate failure risk
                    failure_risk = self._calculate_failure_risk(gpu_id)
                    gpu_metrics['failure_risk'] = failure_risk
                    
                    # Log to CSV
                    self._log_metrics(gpu_metrics)
                    
                    # Check if emergency action needed
                    if failure_risk >= 0.7:  # 70% or higher risk
                        logger.warning(f"HIGH FAILURE RISK ({failure_risk:.2f}) DETECTED FOR GPU {gpu_id}")
                        return failure_risk
                
                # Sleep until next collection interval
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)  # Wait before retry
    
    def _collect_metrics(self) -> List[Dict]:
        """Collect detailed metrics from all available GPUs"""
        metrics_list = []
        
        try:
            # Run nvidia-smi for multiple metrics
            output = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=index,name,driver_version,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total,ecc.errors.corrected,ecc.errors.uncorrected', 
                '--format=csv,noheader,nounits'
            ], universal_newlines=True)
            
            # Process each GPU's data
            for line in output.strip().split('\n'):
                values = [val.strip() for val in line.split(',')]
                
                # Handle power.draw which might include 'W' unit
                power_val = values[4]
                power_numeric = float(power_val.replace('W', '')) if 'W' in power_val else float(power_val)
                
                # Build metrics dictionary
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'gpu_id': int(values[0]),
                    'gpu_name': values[1],
                    'driver_version': values[2],
                    'temperature': float(values[3]),
                    'power_draw': power_numeric,
                    'gpu_utilization': float(values[5]),
                    'memory_utilization': float(values[6]),
                    'memory_used': float(values[7]),
                    'memory_total': float(values[8]),
                    'errors': int(values[9]) + int(values[10]) if values[9] and values[10] else 0,
                }
                
                metrics_list.append(metrics)
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        
        return metrics_list
    
    def _calculate_failure_risk(self, gpu_id: int) -> float:
        """
        Calculate the risk of GPU failure based on collected metrics
        Returns a value between 0.0 (no risk) and 1.0 (imminent failure)
        """
        if gpu_id not in self.metrics_history or not self.metrics_history[gpu_id]:
            return 0.0
        
        history = self.metrics_history[gpu_id]
        current = history[-1]
        
        # Initialize risk factors
        risk_factors = []
        
        # 1. Temperature risk
        temp = current['temperature']
        temp_risk = min(1.0, max(0.0, (temp - 70) / (self.alert_thresholds['temperature'] - 70)))
        risk_factors.append(('temperature', temp_risk))
        
        # 2. Memory utilization risk
        mem_util = current['memory_utilization']
        mem_risk = min(1.0, max(0.0, (mem_util - 85) / (self.alert_thresholds['memory_utilization'] - 85)))
        risk_factors.append(('memory', mem_risk))
        
        # 3. Error detection (any errors are concerning)
        error_risk = 1.0 if current['errors'] > 0 else 0.0
        risk_factors.append(('errors', error_risk))
        
        # 4. Power fluctuation (need at least 2 data points)
        power_risk = 0.0
        if len(history) >= 2:
            prev_power = history[-2]['power_draw']
            curr_power = current['power_draw']
            if prev_power > 0:  # Avoid division by zero
                power_change_pct = abs((curr_power - prev_power) / prev_power * 100)
                power_risk = min(1.0, power_change_pct / self.alert_thresholds['power_fluctuation'])
        risk_factors.append(('power', power_risk))
        
        # 5. GPU utilization drop (indicates potential driver issues)
        util_risk = 0.0
        if len(history) >= 10:  # Need enough history
            baseline_util = np.mean([h['gpu_utilization'] for h in history[:-5]])
            current_util = current['gpu_utilization']
            if baseline_util > 50:  # Only relevant if GPU was being used
                util_drop_pct = max(0, (baseline_util - current_util))
                util_risk = min(1.0, util_drop_pct / self.alert_thresholds['gpu_utilization_drop'])
        risk_factors.append(('utilization_drop', util_risk))
        
        # 6. Rapid temperature increase
        temp_rise_risk = 0.0
        if len(history) >= 6:  # Need enough history for trend
            temps = [h['temperature'] for h in history[-6:]]
            time_window_minutes = (self.interval * 6) / 60  # Convert to minutes
            if time_window_minutes > 0:
                temp_rise_rate = (temps[-1] - temps[0]) / time_window_minutes
                if temp_rise_rate > 0:  # Only consider temperature increases
                    temp_rise_risk = min(1.0, temp_rise_rate / self.alert_thresholds['temperature_rise_rate'])
        risk_factors.append(('temp_rise', temp_rise_risk))
        
        # Combine risk factors with weights
        weights = {
            'temperature': 0.25,
            'memory': 0.15,
            'errors': 0.25,
            'power': 0.10,
            'utilization_drop': 0.15,
            'temp_rise': 0.10
        }
        
        total_risk = sum(weight * risk for name, risk in risk_factors if name in weights)
        
        # Log high-risk situations
        if total_risk > 0.5:
            risk_details = ', '.join([f"{name}: {risk:.2f}" for name, risk in risk_factors])
            logger.warning(f"GPU {gpu_id} failure risk: {total_risk:.2f} ({risk_details})")
        
        return total_risk
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to CSV file"""
        try:
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    metrics['timestamp'],
                    metrics['gpu_id'],
                    metrics['gpu_name'],
                    metrics['driver_version'],
                    metrics['temperature'],
                    metrics['power_draw'],
                    metrics['gpu_utilization'],
                    metrics['memory_utilization'],
                    metrics['memory_used'],
                    metrics['memory_total'],
                    metrics['errors'],
                    metrics['failure_risk']
                ])
        except Exception as e:
            logger.error(f"Error logging metrics to CSV: {e}")
    
    def get_current_failure_risks(self) -> Dict[int, float]:
        """Return current failure risk assessment for all GPUs"""
        risks = {}
        for gpu_id, history in self.metrics_history.items():
            if history:
                risks[gpu_id] = history[-1].get('failure_risk', 0.0)
        return risks


class MegatronEmergencyCheckpointer:
    """
    Emergency checkpointing system for Megatron-LM to handle imminent GPU failures
    """
    
    def __init__(
        self,
        megatron_path: str,
        checkpoint_dir: str,
        config_file: str = None,
        metrics_collector: GPUMetricsCollector = None,
        risk_threshold: float = 0.7,
        monitoring_interval: int = 5
    ):
        self.megatron_path = megatron_path
        self.checkpoint_dir = checkpoint_dir
        self.config_file = config_file
        
        # Use provided metrics collector or create a new one
        self.metrics_collector = metrics_collector or GPUMetricsCollector()
        self.risk_threshold = risk_threshold
        self.monitoring_interval = monitoring_interval
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load Megatron configuration if file provided
        self.config = None
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Most recent checkpointing info
        self.last_checkpoint_time = None
        self.last_checkpoint_path = None
        self.emergency_checkpoint_count = 0
    
    def start_monitoring(self):
        """Start the GPU monitoring and emergency checkpointing system"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        # Start metrics collection if not already running
        if not hasattr(self.metrics_collector, 'running') or not self.metrics_collector.running:
            self.metrics_collector.start_collection()
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Started emergency checkpoint monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        logger.info("Stopped emergency checkpoint monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop to check for GPU failure risks"""
        while self.is_monitoring:
            try:
                # Get current risk assessments
                gpu_risks = self.metrics_collector.get_current_failure_risks()
                
                # Check if any GPU exceeds the risk threshold
                high_risk_gpus = {gpu_id: risk for gpu_id, risk in gpu_risks.items() if risk >= self.risk_threshold}
                
                if high_risk_gpus:
                    # Format risk info for logging
                    risk_info = ', '.join([f"GPU {gpu_id}: {risk:.2f}" for gpu_id, risk in high_risk_gpus.items()])
                    logger.warning(f"High failure risk detected: {risk_info}")
                    
                    # Create emergency checkpoint 
                    self._trigger_emergency_checkpoint(high_risk_gpus)
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retry
    
    def _trigger_emergency_checkpoint(self, high_risk_gpus: Dict[int, float]):
        """
        Trigger an emergency checkpoint process to save training state
        
        Args:
            high_risk_gpus: Dictionary of {gpu_id: risk_score} for high-risk GPUs
        """
        # Don't create checkpoints too frequently
        current_time = time.time()
        if (self.last_checkpoint_time and 
            (current_time - self.last_checkpoint_time) < 300):  # 5 minutes
            logger.info("Skipping checkpoint - too soon since last emergency checkpoint")
            return
        
        # Create a timestamped directory for this emergency checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.emergency_checkpoint_count += 1
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"emergency_checkpoint_{timestamp}_risk{max(high_risk_gpus.values()):.2f}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        
        try:
            # Create checkpoint metadata
            metadata = {
                'timestamp': timestamp,
                'high_risk_gpus': high_risk_gpus,
                'checkpoint_reason': 'Imminent GPU failure predicted',
                'emergency_checkpoint_id': self.emergency_checkpoint_count
            }
            
            with open(os.path.join(checkpoint_path, 'emergency_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Execute Megatron's checkpointing logic
            # In a production environment, you would use Megatron's actual API
            # or the appropriate signaling mechanism to trigger checkpointing
            self._execute_megatron_checkpoint(checkpoint_path)
            
            # Update checkpoint info
            self.last_checkpoint_time = current_time
            self.last_checkpoint_path = checkpoint_path
            
            logger.info(f"Emergency checkpoint created: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to create emergency checkpoint: {e}")
    
    def _execute_megatron_checkpoint(self, checkpoint_path: str):
        """
        Execute the actual Megatron checkpointing process
        
        In a real implementation, this would integrate with Megatron's
        checkpointing API or mechanism rather than using a subprocess
        """
        # This is a placeholder for integration with Megatron's actual checkpoint mechanism
        logger.info(f"Signaling Megatron to create checkpoint at: {checkpoint_path}")
        
        # For actual implementation, you would:
        # 1. Either call into Megatron's Python API to trigger a checkpoint
        # 2. Or send a signal to the Megatron process to initiate checkpointing
        # 3. Or write to a shared file/socket that Megatron monitors
        
        # Example of how you might signal via a file
        with open(os.path.join(checkpoint_path, 'CHECKPOINT_NOW'), 'w') as f:
            f.write(f"Emergency checkpoint triggered at {datetime.now().isoformat()}")
        
        # In a real implementation, you would wait for confirmation that the
        # checkpoint completed successfully
        
        # Simulate checkpoint process for demonstration
        logger.info("Simulated checkpoint process complete")
        return True
    
    def manual_checkpoint(self):
        """Manually trigger an emergency checkpoint"""
        logger.info("Manual emergency checkpoint triggered")
        self._trigger_emergency_checkpoint({0: 1.0})  # Force checkpoint with max risk
        return self.last_checkpoint_path

# Example usage
if __name__ == "__main__":
    # Initialize the metrics collector
    metrics_collector = GPUMetricsCollector(interval=5)
    
    # Initialize the emergency checkpointer
    checkpointer = MegatronEmergencyCheckpointer(
        megatron_path="/path/to/Megatron-LM",
        checkpoint_dir="./emergency_checkpoints",
        metrics_collector=metrics_collector,
        risk_threshold=0.7
    )
    
    # Start monitoring
    checkpointer.start_monitoring()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        checkpointer.stop_monitoring()