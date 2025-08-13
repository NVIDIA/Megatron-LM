from tests.unit_tests.test_utilities import Utils
import torch
from megatron.core.unified_timer_event import TimerContext, TimerEvent


def test_unified_timer_event():
    Utils.initialize_model_parallel(1, 1)
    print(f"Testing TimerEvent")
    
    # Basic timing test
    start = TimerEvent(enable_timing=True)
    end = TimerEvent(enable_timing=True)
    
    start.record()
    
    # Simulate some work
    x = torch.randn(1000, 1000)
    y = torch.mm(x, x.t())
    torch.sum(y)
    
    end.record()
    end.synchronize()
    
    elapsed = start.elapsed_time(end)
    print(f"Elapsed time: {elapsed:.4f} ms")
    
    # Test context manager
    print("\nTesting context manager:")
    with TimerContext() as timer:
        x = torch.randn(500, 500)
        y = torch.mm(x, x.t())
        result = torch.sum(y)
    
    print(f"Context manager elapsed: {timer.elapsed_time:.4f} ms")
    Utils.destroy_model_parallel()