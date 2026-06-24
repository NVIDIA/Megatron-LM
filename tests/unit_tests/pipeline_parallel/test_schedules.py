# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.transformer.cuda_graphs import (
    convert_schedule_table_to_order,
    get_overlap_moe_expert_parallel_comm_order,
)
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def _populate_embedding_and_position_groups(pp_group):
    """Create *new* embedding-related process groups from *pp_group* ranks."""

    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))

    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    return pos_embd_pg, embd_pg


def test_get_forward_backward_func():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=4,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()


def test_deallocate_output_tensor():
    out = torch.tensor([[1, 2, 3], [4, 5, 6]])
    schedule.deallocate_output_tensor(out)
    assert out.nelement() == 6


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "pipeline_model_parallel_size,microbatch_group_size_per_vp_stage",
    [(1, 1), (2, 2), (2, 4), (4, 4), (4, 5), (8, 9), (8, 11)],
)
@pytest.mark.parametrize("num_microbatches", [8, 32])
@pytest.mark.parametrize("virtual_pipeline_model_parallel_size", [None, 2, 4, 8])
def test_get_pipeline_parallel_order(
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    num_microbatches,
    microbatch_group_size_per_vp_stage,
):
    if pipeline_model_parallel_size == 1 and virtual_pipeline_model_parallel_size is not None:
        return

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    num_model_chunks = (
        virtual_pipeline_model_parallel_size
        if virtual_pipeline_model_parallel_size is not None
        else 1
    )

    _, _, num_warmup_microbatches, _ = schedule.get_pp_rank_microbatches(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, False
    )
    schedule_table = schedule.get_schedule_table(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage
    )
    order = convert_schedule_table_to_order(
        num_warmup_microbatches, num_model_chunks, schedule_table
    )

    assert max(order) == num_model_chunks
    assert len(order) == num_microbatches * num_model_chunks * 2
    order_cnt = {}
    accumulated_order = 0
    for o in order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0
    assert 0 not in order_cnt
    for k, v in order_cnt.items():
        assert -k in order_cnt and order_cnt[-k] == v

    layers_per_chunk = 2
    num_layers_per_chunk = [layers_per_chunk] * num_model_chunks
    # disable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, False
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk
    assert len(chunk_id_list) == len(overlapped_order)
    order_cnt = {}
    accumulated_order = 0
    for o in overlapped_order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0

    # enable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, True
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk * 3 // 2
    assert len(chunk_id_list) == len(overlapped_order)
    from math import ceil

    order_cnt = {}
    accumulated_order = 0
    prev_o = 0
    for o in overlapped_order:
        if ceil(o) != o:
            assert prev_o - 0.5 == o
        else:
            order_cnt[o] = order_cnt.get(o, 0) + 1
            if o < 0:
                assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
            elif -o in order_cnt:
                assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        prev_o = o
    assert accumulated_order < 0

    Utils.destroy_model_parallel()


def test_forward_backward_func_without_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 100),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


def test_forward_backward_func_with_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=None,
        model=[model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]
    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_uneven_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model_a = torch.nn.Linear(4, 1)
    model_b = torch.nn.Linear(8, 1)
    model_a.vp_stage = 0
    model_b.vp_stage = 1

    def set_input_tensor(input_tensor):
        return None

    model_a.set_input_tensor = set_input_tensor
    model_b.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model_a.config = config
    model_b.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model_a.model_type = ModelType.encoder_or_decoder
        model_b.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_without_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_without_interleaving produces the same output
    with and without explicit process group parameters."""

    # Initialize model parallel with pipeline parallelism (no interleaving)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def dummy_step_func(data_iterator, model):
        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    # Create model
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def return_none(input_tensor):
        return None

    model.set_input_tensor = return_none

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    config.finalize_model_grads_func = finalize_model_grads
    model.config = config

    # Mock custom_backward to avoid actual computation
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    # Common arguments for both calls
    common_args = {
        'forward_step_func': dummy_step_func,
        'data_iterator': None,
        'model': [model],
        'num_microbatches': micro_batch_size,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': True,
    }

    # First call: without providing process group parameters (they'll be created internally)
    losses_reduced_default = schedule.forward_backward_pipelining_without_interleaving(
        **common_args
    )

    grid = HyperCommGrid([2, 1, 4, 1], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = dp_cp_group
    pg_collection.cp = grid.create_pg("cp")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=p2p_communicator, pg_collection=pg_collection, **common_args
    )

    assert len(losses_reduced_default) == len(
        losses_reduced_explicit
    ), "Output lengths should be identical"

    for i, (default_loss, explicit_loss) in enumerate(
        zip(losses_reduced_default, losses_reduced_explicit)
    ):
        assert (
            default_loss == explicit_loss
        ), f"Loss at index {i} should be identical between default and explicit PG calls"
    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_with_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_with_interleaving produces the same output
    with and without explicit process group parameters."""

    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    grid = HyperCommGrid([1, 1, 4, 2], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.cp = grid.create_pg("cp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = grid.create_pg(["dp", "cp"])

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
        pg_collection=pg_collection,
        p2p_communicator=p2p_communicator,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


def test_forward_backward_no_pipelining_with_custom_pgs(mocker):
    """Validate no-pipeline schedule when explicit custom PGs are provided."""

    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank_local = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank_local, {'loss_reduced': rank_local}

        dummy_inp = torch.ones(1, 4)
        return model(dummy_inp), loss_func

    # Simple model.
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'
    model.set_input_tensor = lambda _tensor: None  # type: ignore[assignment]

    # Minimal config.
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    grid = HyperCommGrid([2, 1, 1, 4], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = tp_group
    pg_collection.cp = cp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.pp = pp_group
    pg_collection.dp_cp = dp_cp_group

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 10),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
        pg_collection=pg_collection,
    )

    expected = {'loss_reduced': Utils.rank}
    for l in losses_reduced:
        assert l['loss_reduced'] == expected['loss_reduced']

    Utils.destroy_model_parallel()


class _NonInterleavedPipelineTestModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = 'unit-test'
        self.config = config
        self.input_tensor = None
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False).cuda()
        torch.nn.init.constant_(self.proj.weight, 0.01)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    def forward(self):
        if self.input_tensor is None:
            x = torch.ones(
                512,
                8,
                self.config.hidden_size,
                device='cuda',
                dtype=self.config.pipeline_dtype,
                requires_grad=True,
            )
        else:
            x = self.input_tensor
        return self.proj(x)


def _make_non_interleaved_test_model(config):
    return _NonInterleavedPipelineTestModel(config)


def _make_non_interleaved_test_config(**overrides):
    defaults = dict(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        batch_p2p_comm=False,
    )
    defaults.update(overrides)
    config = ModelParallelConfig(**defaults)
    config.hidden_size = 256
    return config


def _make_non_interleaved_forward_step_func():
    def forward_step_func(data_iterator, model):
        output_tensor = model()

        def loss_func(loss_tensor):
            reduced = loss_tensor.sum()
            return reduced, {'loss_reduced': reduced.detach().clone()}

        return output_tensor, loss_func

    return forward_step_func


@pytest.mark.internal
class TestNonInterleavedOverlapGuards:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=4
        )

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _call_kwargs(self, model, **overrides):
        defaults = dict(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[model],
            num_microbatches=8,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )
        defaults.update(overrides)
        return defaults

    def test_rejects_forward_only(self):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True)
        model = _make_non_interleaved_test_model(config)
        with pytest.raises(NotImplementedError, match="forward_only"):
            schedule.forward_backward_pipelining_without_interleaving(
                **self._call_kwargs(model, forward_only=True)
            )

    def test_rejects_adjust_tensor_shapes_fn(self):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True)
        model = _make_non_interleaved_test_model(config)
        with pytest.raises(NotImplementedError, match="adjust_tensor_shapes_fn"):
            schedule.forward_backward_pipelining_without_interleaving(
                **self._call_kwargs(model, adjust_tensor_shapes_fn=lambda r, s: (r, s))
            )

    def test_rejects_multimodule(self):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True)
        model = _make_non_interleaved_test_model(config)
        multimodule_pg_collection = MultiModuleProcessGroupCollection(
            module_pgs={"llm": ProcessGroupCollection()}, language_model_module_name="llm"
        )
        with pytest.raises(NotImplementedError, match="multimodule"):
            schedule.forward_backward_pipelining_without_interleaving(
                **self._call_kwargs(model, pg_collection=multimodule_pg_collection)
            )

    def test_rejects_batch_p2p_comm(self):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True, batch_p2p_comm=True)
        model = _make_non_interleaved_test_model(config)
        with pytest.raises(ValueError, match="batch_p2p_comm"):
            schedule.forward_backward_pipelining_without_interleaving(**self._call_kwargs(model))

    def test_rejects_ring_exchange(self):
        config = _make_non_interleaved_test_config(
            overlap_p2p_comm=True, use_ring_exchange_p2p=True
        )
        model = _make_non_interleaved_test_model(config)
        with pytest.raises(NotImplementedError, match="use_ring_exchange_p2p"):
            schedule.forward_backward_pipelining_without_interleaving(**self._call_kwargs(model))

    def test_rejects_warmup_flush(self):
        config = _make_non_interleaved_test_config(
            overlap_p2p_comm=True, overlap_p2p_comm_warmup_flush=True
        )
        model = _make_non_interleaved_test_model(config)
        with pytest.raises(NotImplementedError, match="overlap_p2p_comm_warmup_flush"):
            schedule.forward_backward_pipelining_without_interleaving(**self._call_kwargs(model))

    def test_rejects_variable_seq_lengths(self):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True)
        config.variable_seq_lengths = True
        model = _make_non_interleaved_test_model(config)
        with pytest.raises(NotImplementedError, match="variable_seq_lengths"):
            schedule.forward_backward_pipelining_without_interleaving(**self._call_kwargs(model))

    def test_rejects_mtp_standalone(self):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True)
        config.mtp_standalone = True
        model = _make_non_interleaved_test_model(config)
        with pytest.raises(NotImplementedError, match="mtp_standalone"):
            schedule.forward_backward_pipelining_without_interleaving(**self._call_kwargs(model))

    def test_rejects_shape_count_mismatch(self, mocker):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True)
        model = _make_non_interleaved_test_model(config)
        mocker.patch(
            "megatron.core.pipeline_parallel.schedules.get_tensor_shapes",
            side_effect=[[(512, 8, 256)], [(512, 8, 256), (512, 8, 256)]],
        )
        with pytest.raises(NotImplementedError, match="matching recv/send tensor shape counts"):
            schedule.forward_backward_pipelining_without_interleaving(**self._call_kwargs(model))


@pytest.mark.internal
class TestNonInterleavedOverlapExecution:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=4
        )

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _run_overlap_vs_baseline_training(self, deallocate_pipeline_outputs):
        baseline_config = _make_non_interleaved_test_config(
            overlap_p2p_comm=False, deallocate_pipeline_outputs=deallocate_pipeline_outputs
        )
        baseline_model = _make_non_interleaved_test_model(baseline_config)

        overlap_config = _make_non_interleaved_test_config(
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            deallocate_pipeline_outputs=deallocate_pipeline_outputs,
        )
        overlap_model = _make_non_interleaved_test_model(overlap_config)
        overlap_model.load_state_dict(baseline_model.state_dict())

        baseline_losses = schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[baseline_model],
            num_microbatches=8,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )

        overlap_losses = schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[overlap_model],
            num_microbatches=8,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )

        assert isinstance(baseline_losses, list)
        assert isinstance(overlap_losses, list)
        assert len(baseline_losses) == len(overlap_losses)
        for baseline_loss, overlap_loss in zip(baseline_losses, overlap_losses):
            assert baseline_loss.keys() == overlap_loss.keys()
            assert torch.equal(baseline_loss['loss_reduced'], overlap_loss['loss_reduced'])

    def _run_overlap_vs_baseline_gradients(self, deallocate_pipeline_outputs):
        baseline_config = _make_non_interleaved_test_config(
            overlap_p2p_comm=False, deallocate_pipeline_outputs=deallocate_pipeline_outputs
        )
        baseline_model = _make_non_interleaved_test_model(baseline_config)

        overlap_config = _make_non_interleaved_test_config(
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            deallocate_pipeline_outputs=deallocate_pipeline_outputs,
        )
        overlap_model = _make_non_interleaved_test_model(overlap_config)
        overlap_model.load_state_dict(baseline_model.state_dict())

        schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[baseline_model],
            num_microbatches=8,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )

        schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[overlap_model],
            num_microbatches=8,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )

        baseline_grads = {}
        for name, param in baseline_model.named_parameters():
            assert param.grad is not None, f"Expected baseline gradient for {name}"
            baseline_grads[name] = param.grad.detach().clone()

        overlap_grads = {}
        for name, param in overlap_model.named_parameters():
            assert param.grad is not None, f"Expected overlap gradient for {name}"
            overlap_grads[name] = param.grad.detach().clone()

        assert baseline_grads.keys() == overlap_grads.keys()
        for name in baseline_grads:
            assert torch.allclose(
                baseline_grads[name], overlap_grads[name]
            ), f"Gradient mismatch for {name}"

    def test_overlap_matches_baseline_training(self):
        self._run_overlap_vs_baseline_training(deallocate_pipeline_outputs=False)

    def test_overlap_matches_baseline_training_with_deallocate_pipeline_outputs(self):
        self._run_overlap_vs_baseline_training(deallocate_pipeline_outputs=True)

    def test_overlap_matches_baseline_gradients(self):
        self._run_overlap_vs_baseline_gradients(deallocate_pipeline_outputs=False)

    def test_overlap_matches_baseline_gradients_with_deallocate_pipeline_outputs(self):
        self._run_overlap_vs_baseline_gradients(deallocate_pipeline_outputs=True)

    def test_wait_send_next_before_deallocate(self, mocker):
        from megatron.core.pipeline_parallel.p2p_communication import P2PAsyncHandleSet

        events = []
        tracked_pairs = []
        waited_output_seqs = set()
        original_deallocate = schedule.deallocate_output_tensor
        original_wait_send_next = P2PAsyncHandleSet.wait_send_next
        original_send_forward_recv_backward = P2PCommunicator.send_forward_recv_backward

        def _find_output_entry(output_tensor):
            for entry in tracked_pairs:
                if entry["output_tensor"] is output_tensor:
                    return entry
            return None

        def _find_handle_entry(handle_set):
            for entry in tracked_pairs:
                if entry["handle_set"] is handle_set:
                    return entry
            return None

        def tracking_send_forward_recv_backward(
            self, output_tensors, tensor_shapes, is_last_stage, overlap_p2p_comm=False
        ):
            result = original_send_forward_recv_backward(
                self, output_tensors, tensor_shapes, is_last_stage, overlap_p2p_comm
            )
            if overlap_p2p_comm:
                _, handle_set = result
                tracked_pairs.append(
                    {
                        "seq": len(tracked_pairs),
                        "output_tensor": output_tensors,
                        "handle_set": handle_set,
                    }
                )
            return result

        def tracking_deallocate(out, deallocate=False):
            entry = _find_output_entry(out)
            if deallocate and entry is not None:
                events.append(("deallocate", entry["seq"]))
            return original_deallocate(out, deallocate)

        def tracking_wait_send_next(self):
            entry = _find_handle_entry(self)
            if entry is not None and entry["seq"] not in waited_output_seqs:
                waited_output_seqs.add(entry["seq"])
                events.append(("wait_send_next", entry["seq"]))
            return original_wait_send_next(self)

        mocker.patch.object(
            P2PCommunicator, "send_forward_recv_backward", tracking_send_forward_recv_backward
        )
        mocker.patch(
            "megatron.core.pipeline_parallel.schedules.deallocate_output_tensor",
            side_effect=tracking_deallocate,
        )
        mocker.patch.object(P2PAsyncHandleSet, "wait_send_next", tracking_wait_send_next)

        config = _make_non_interleaved_test_config(
            overlap_p2p_comm=True, batch_p2p_comm=False, deallocate_pipeline_outputs=True
        )
        model = _make_non_interleaved_test_model(config)

        schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[model],
            num_microbatches=8,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )

        deallocate_events = [event for event in events if event[0] == "deallocate"]
        assert deallocate_events, "Expected at least one steady-state overlap deallocation event."
        for deallocate_event in deallocate_events:
            wait_event = ("wait_send_next", deallocate_event[1])
            assert wait_event in events, (
                f"Missing wait_send_next for overlap output seq={deallocate_event[1]}. "
                f"Events: {events}"
            )
            wait_index = events.index(wait_event)
            deallocate_index = events.index(deallocate_event)
            assert wait_index <= deallocate_index, (
                f"wait_send_next (at {wait_index}) must happen before or at "
                f"deallocate (at {deallocate_index}) for output seq={deallocate_event[1]}. "
                f"Events: {events}"
            )

    def test_all_warmup_no_steady_state(self):
        config = _make_non_interleaved_test_config(overlap_p2p_comm=True, batch_p2p_comm=False)
        model = _make_non_interleaved_test_model(config)

        losses = schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[model],
            num_microbatches=1,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )

        assert isinstance(losses, list)

    def test_single_steady_state_completes_and_waits_send_handles(self, mocker):
        from megatron.core.pipeline_parallel.p2p_communication import P2PAsyncHandleSet

        wait_events = []
        original_wait_send_next = P2PAsyncHandleSet.wait_send_next
        original_wait_send_prev = P2PAsyncHandleSet.wait_send_prev

        def tracking_wait_send_next(self):
            wait_events.append("wait_send_next")
            return original_wait_send_next(self)

        def tracking_wait_send_prev(self):
            wait_events.append("wait_send_prev")
            return original_wait_send_prev(self)

        mocker.patch.object(P2PAsyncHandleSet, "wait_send_next", tracking_wait_send_next)
        mocker.patch.object(P2PAsyncHandleSet, "wait_send_prev", tracking_wait_send_prev)

        config = _make_non_interleaved_test_config(overlap_p2p_comm=True, batch_p2p_comm=False)
        model = _make_non_interleaved_test_model(config)

        losses = schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=_make_non_interleaved_forward_step_func(),
            data_iterator=None,
            model=[model],
            num_microbatches=4,
            seq_length=512,
            micro_batch_size=8,
            forward_only=False,
        )

        assert isinstance(losses, list)
        assert (
            wait_events
        ), "Expected at least one send-handle wait in the single steady-state case."
