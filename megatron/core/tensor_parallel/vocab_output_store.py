
class VocabOutputStore:
    """
    For storing and retrieving intermediate results of the VocabParallelOutput layer.
    """

    microbatch_id = 0
    forward_cache = []
    backward_cache = []

    @classmethod
    def forward_store(cls, sum_exp_logits, logits_max, predicted_logits, target_mask,
                      softmax_grad_input, ground_truth_grad_input):
        while len(cls.forward_cache) <= cls.microbatch_id:
            cls.forward_cache.append(None)
        cls.forward_cache[cls.microbatch_id] = (
            sum_exp_logits, logits_max, predicted_logits, target_mask, softmax_grad_input, ground_truth_grad_input
        )

    @classmethod
    def forward_get(cls):
        contents = cls.forward_cache[cls.microbatch_id]
        cls.forward_cache[cls.microbatch_id] = None
        return contents

    @classmethod
    def backward_store(cls, sum_exp_logits, logits_max, grad_output):
        while len(cls.backward_cache) <= cls.microbatch_id:
            cls.backward_cache.append(None)
        cls.backward_cache[cls.microbatch_id] = (
            sum_exp_logits, logits_max, grad_output
        )

    @classmethod
    def backward_get(cls):
        contents = cls.backward_cache[cls.microbatch_id]
        cls.backward_cache[cls.microbatch_id] = None
        return contents
