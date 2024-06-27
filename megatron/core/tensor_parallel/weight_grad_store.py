import queue

class WeightGradStore:

    cache = []
    weight_grad_queue = queue.Queue()
    combine_bw = True

    @classmethod
    def set_combine_bw(cls, combine_bw):
        # For the following backward pass, combine W with B and skip next W.
        cls.combine_bw = combine_bw

    @classmethod
    def put(cls, total_input, grad_output, weight, func):
        if cls.combine_bw == True:
            func(total_input, grad_output, weight)
            return
        # Store the weight gradient computation of linear layers.
        cls.cache.append((total_input, grad_output, weight, func))

    @classmethod
    def flush(cls):
        # Collect all stored computations during backward as a W.
        cls.weight_grad_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls):
        # Execute a single W.
        assert cls.weight_grad_queue.qsize() > 0
        stored_grads = cls.weight_grad_queue.get()
        for total_input, grad_output, weight, func in stored_grads:
            func(total_input, grad_output, weight)