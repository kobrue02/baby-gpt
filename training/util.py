def get_mps_memory_info():
    """Helper function to monitor MPS memory usage"""
    import torch

    if torch.backends.mps.is_available():
        return {
            "allocated": torch.mps.current_allocated_memory() / 1024**3,  # GB
            "reserved": torch.mps.driver_allocated_memory() / 1024**3,  # GB
        }
    return None


def cleanup_mps_memory():
    """Clean up MPS memory between training steps if needed"""
    import torch
    import gc

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        gc.collect()