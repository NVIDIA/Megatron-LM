import time
import torch
from typing import Optional

from megatron.core.device_utils import get_xla_model

xm = get_xla_model()


class TimerEvent:
    """
    A unified timer event class that works with both CUDA and PyTorch/XLA backends.
    
    This class provides a common interface similar to torch.cuda.Event for timing
    operations across different PyTorch backends.
    
    Args:
        enable_timing (bool): Whether to enable timing functionality
        blocking (bool): Whether synchronization should be blocking (XLA always blocks)
    
    Example:
        # Works with both CUDA and XLA
        start_event = TimerEvent(enable_timing=True)
        end_event = TimerEvent(enable_timing=True)
        
        start_event.record()
        # ... your operations ...
        end_event.record()
        
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"Elapsed time: {elapsed_time:.2f} ms")
    """
    
    def __init__(self, enable_timing: bool = False, blocking: bool = False):
        self.enable_timing = enable_timing
        self.blocking = blocking
        
        # Determine backend
        self.is_cuda = torch.cuda.is_available()
        self.is_xla = xm is not None
        
        # Initialize backend-specific components
        if self.is_cuda:
            self._cuda_event = torch.cuda.Event(enable_timing=enable_timing, 
                                              blocking=blocking)
            self._recorded_time = None
        elif self.is_xla:
            self._recorded_time = None
            self._xla_step_recorded = False
        else:
            # CPU fallback
            self._recorded_time = None
    
    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Record the event on the current stream.
        
        Args:
            stream: CUDA stream to record on (ignored for XLA)
        """
        if not self.enable_timing:
            return
            
        if self.is_cuda:
            if stream is not None:
                self._cuda_event.record(stream)
            else:
                self._cuda_event.record()
        elif self.is_xla:
            # For XLA, we mark the step and record the time
            xm.mark_step()  # Ensure operations are queued
            if self.blocking:
                xm.wait_device_ops()  # Wait for completion
            self._recorded_time = time.perf_counter()
            self._xla_step_recorded = True
        else:
            # CPU fallback
            self._recorded_time = time.perf_counter()
    
    def synchronize(self) -> None:
        """
        Wait for the event to complete.
        """
        if self.is_cuda:
            self._cuda_event.synchronize()
        elif self.is_xla and self._xla_step_recorded:
            xm.wait_device_ops()
        # CPU operations are already synchronous
    
    def elapsed_time(self, end_event: 'TimerEvent') -> float:
        """
        Calculate elapsed time between this event and another event.
        
        Args:
            end_event: The end event to measure time to
            
        Returns:
            Elapsed time in milliseconds
            
        Raises:
            RuntimeError: If timing is not enabled or events not recorded
        """
        if not self.enable_timing or not end_event.enable_timing:
            raise RuntimeError("Events must have enable_timing=True")
        
        if self.is_cuda and end_event.is_cuda:
            return self._cuda_event.elapsed_time(end_event._cuda_event)
        elif self.is_xla and end_event.is_xla:
            if self._recorded_time is None or end_event._recorded_time is None:
                raise RuntimeError("Both events must be recorded")
            # Ensure all operations are complete
            xm.wait_device_ops()
            # Convert to milliseconds (perf_counter returns seconds)
            return (end_event._recorded_time - self._recorded_time) * 1000.0
        elif not (self.is_cuda or self.is_xla) and not (end_event.is_cuda or end_event.is_xla):
            # CPU fallback
            if self._recorded_time is None or end_event._recorded_time is None:
                raise RuntimeError("Both events must be recorded")
            return (end_event._recorded_time - self._recorded_time) * 1000.0
        else:
            raise RuntimeError("Cannot measure elapsed time between different backends")
    
    def query(self) -> bool:
        """
        Check if the event has completed.
        
        Returns:
            True if the event has completed, False otherwise
        """
        if self.is_cuda:
            return self._cuda_event.query()
        elif self.is_xla:
            # XLA operations are typically blocking, so if recorded, it's complete
            return self._xla_step_recorded
        else:
            # CPU operations are synchronous
            return self._recorded_time is not None
    
    def wait(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Make the given stream wait for this event.
        
        Args:
            stream: CUDA stream that should wait (ignored for XLA/CPU)
        """
        if self.is_cuda and stream is not None:
            self._cuda_event.wait(stream)
        else:
            # For XLA and CPU, just synchronize
            self.synchronize()


class TimerContext:
    """
    Context manager for easy timing of code blocks.
    
    Example:
        with TimerContext() as timer:
            # ... your operations ...
            pass
        print(f"Elapsed: {timer.elapsed_time:.2f} ms")
    """
    
    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_event = TimerEvent(enable_timing=True)
        self.end_event = TimerEvent(enable_timing=True)
        self.start_event.record()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
