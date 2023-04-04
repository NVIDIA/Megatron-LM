from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
import torch
from tests.unit_tests.test_utilities import Utils
import numpy as np

def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4,2)
    vocab_parallel_logits = torch.range(0,7).repeat(16,4).cuda()
    target = torch.arange(0,32,2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = torch.tensor([10.2309,  8.2309,  6.2309,  4.2309, 10.2309,  8.2309,  6.2309,  4.2309,
        10.2309,  8.2309,  6.2309,  4.2309, 10.2309,  8.2309,  6.2309,  4.2309]).cuda()
    assert(torch.equal(torch.round(expected_output), torch.round(output)))
    Utils.destroy_model_parallel()