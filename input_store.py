from megatron.core import mpu
from megatron.training import get_args

class InputStore:
    """
    For storing and retrieving batch input that are partially unused.
    """

    cache = []

    @classmethod
    def save_batch(cls, microbatch_id, data):
        while len(cls.cache) <= microbatch_id:
            cls.cache.append(None)
        cls.cache[microbatch_id] = data

    @classmethod
    def get_batch(cls, microbatch_id):
        contents = cls.cache[microbatch_id]
        if (
            mpu.get_virtual_vocab_parallel_chunk() == 3
        ):
            cls.cache[microbatch_id] = None
        elif (
            ((not mpu.is_pipeline_last_stage()) or (get_args().use_interlaced_schedule))
            and (mpu.get_virtual_vocab_parallel_chunk() == 1)
        ):
            cls.cache[microbatch_id] = None
        return contents
