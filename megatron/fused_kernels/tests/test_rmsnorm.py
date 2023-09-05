from megatron.model.fused_layer_norm import MixedFusedRMSNorm
from .rmsnorm import RMSNorm


def test_load_fused_kernels():
    try:
        import fused_layer_norm_cuda

        print("[Success] load_fused_kernels")
    except ImportError as e:
        print("[Fail] load_fused_kernels")
        raise e


def test_rms_norm():
    from transformers import BertTokenizer
    from transformers.models.bert.modeling_bert import BertModel

    bert = BertModel.from_pretrained("bert-base-cased").cuda().half()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    test_text = (
        "Hello. How are you? I am fine thank you and you? yes Good. "
        "hi hi hi hi hi hi hi hi hi hi hi hi hi"  # 32
    )

    tokens = tokenizer(
        [test_text] * 4,
        return_tensors="pt",
    )

    # [bsz, seq_len, d_model]
    embedding_output = (
        bert.embeddings(
            input_ids=tokens["input_ids"].cuda(),
            position_ids=None,
            token_type_ids=tokens["token_type_ids"].cuda(),
            inputs_embeds=None,
            past_key_values_length=0,
        )
        .cuda()
        .half()
    )

    fused_rmsnorm_layer = (
        MixedFusedRMSNorm(normalized_shape=embedding_output.size(-1)).cuda().half()
    )

    rmsnorm_layer = (
        RMSNorm(dim=embedding_output.size(-1)).cuda().half()
    )

    fused_output = fused_rmsnorm_layer(embedding_output)
    torch_output = rmsnorm_layer(embedding_output)
    test_result = (fused_output - torch_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    report = f'''
        > mean_difference={diff}
        > fused_values={fused_output[-1][-1][:5].tolist()}
        > torch_values={torch_output[-1][-1][:5].tolist()}
    '''
    assert diff <= 1e-3, report
