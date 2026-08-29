import torch


def test_post_consumes_comb_transposed(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention.hca import HyperConnection

    x = torch.zeros(1, 2)
    residual = torch.eye(2).unsqueeze(0)
    post = torch.zeros(1, 2)
    comb = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    actual = HyperConnection.post(x, residual, post, comb)

    torch.testing.assert_close(actual, comb.transpose(-1, -2))
