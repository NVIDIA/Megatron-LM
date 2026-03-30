try:
    from .energon_avlm_task_encoder import VisionAudioQASample
    __all__ = ["VisionAudioQASample"]
except (ImportError, ModuleNotFoundError):
    pass
