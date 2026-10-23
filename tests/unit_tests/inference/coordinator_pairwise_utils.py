# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real-model DP routing fixtures; observations never replace production behavior."""

import asyncio
import copy
import queue
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import msgpack
import torch
import zmq

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.engines.async_zmq_communicator import AsyncZMQCommunicator
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.inference.engines import test_dynamic_engine as engine_tests
from tests.unit_tests.inference.test_data_parallel_inference_coordinator import DummyTokenizer
from tests.unit_tests.test_utilities import Utils


async def until(predicate, timeout=30):
    """Wait for an observed state, not for a presumed scheduling delay."""

    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


class CoordinatorThread:
    """Run the unmodified blocking coordinator in its own socket-owning thread."""

    def __init__(self, dp_size, tokenizer, *, setup=None, **options):
        self.events = []
        self.errors = []
        self.addresses = queue.Queue()
        self.ready = threading.Event()
        self.coordinator = None
        self.thread = threading.Thread(
            target=self._run, args=(dp_size, tokenizer, setup, options), daemon=True
        )
        self.thread.start()
        self.address = self.addresses.get(timeout=30)

    def _run(self, dp_size, tokenizer, setup, options):
        try:
            coordinator = DataParallelInferenceCoordinator(
                pipe_connection=SimpleNamespace(send=self.addresses.put, close=lambda: None),
                data_parallel_size=dp_size,
                tokenizer=tokenizer,
                max_requests=8,
                deterministic_mode=True,
                hostname="127.0.0.1",
                **options,
            )
            self.coordinator = coordinator
            coordinator.router_socket.setsockopt(zmq.LINGER, 0)
            if setup is not None:
                setup(coordinator)
            for header, handler in list(coordinator._handlers.items()):

                def observe(c, sender, metadata, bodies, handler=handler, header=header):
                    before = dict(c.request_id_to_rank)
                    result = handler(c, sender, metadata, bodies)
                    self.events.append(
                        dict(
                            header=header,
                            sender=sender,
                            metadata=metadata,
                            before=before,
                            after=dict(c.request_id_to_rank),
                            pending=c._pending_counts.tolist(),
                        )
                    )
                    return result

                coordinator._handlers[header] = observe
            self.ready.set()
            coordinator.start()
        except BaseException as error:
            self.errors.append(error)
        finally:
            if self.coordinator is not None:
                self.coordinator.router_socket.disable_monitor()
                self.coordinator.stop()

    def assert_requests_retired(self):
        c = self.coordinator
        assert not self.errors, self.errors
        for name in (
            "request_id_to_client_id",
            "request_id_to_client_request_id",
            "client_request_to_request_id",
            "request_id_to_rank",
        ):
            assert not getattr(c, name), name
        assert not getattr(c, "request_id_to_sampling_params", {})
        assert not c._pending_counts.any()
        assert self.thread.is_alive(), "Ordinary completion must not stop the service"


class RoutedModel:
    """One real engine per model-parallel group plus rank-zero client controls."""

    def __init__(self, config, engine_factory=None, **coordinator_options):
        self.config = config
        self.rank = torch.distributed.get_rank()
        self.dp_size = Utils.world_size // (
            config.tensor_model_parallel_size * config.pipeline_model_parallel_size
        )
        assert self.dp_size >= 2, "Routing rows require at least two DP replicas"
        self.engine = (
            engine_factory(config)
            if engine_factory is not None
            else engine_tests.DynamicInferenceEngineTestBase._build_test_env(config).engine
        )
        self.tokenizer = DummyTokenizer(vocab_size=config.vocab_size, eod=-1)
        self.engine.controller.tokenizer = self.tokenizer
        self.coordinator_options = coordinator_options
        self.clients = []
        self.witnesses = []
        self.service = None
        self.sync_context = zmq.Context()
        self.sync = AsyncZMQCommunicator(
            self.sync_context, process_group=None, hostname="127.0.0.1"
        )

    async def barrier(self):
        await asyncio.wait_for(self.sync.all_reduce_max(1), timeout=60)

    async def direct(self, prompt, params):
        """Same weights, topology and one-request schedule, without routing."""
        future = self.engine.add_request(10001, prompt, copy.deepcopy(params))
        while self.engine.has_unfinished_requests():
            await self.engine.async_step()
        result = msgpack.unpackb(
            msgpack.packb((await future).serialize(), use_bin_type=True), raw=False
        )
        self.engine.reset()
        return result

    async def start(self):
        if self.rank == 0:
            self.service = CoordinatorThread(
                self.dp_size,
                self.tokenizer,
                block_size_tokens=self.config.context_block_size_tokens,
                enable_prefix_caching=self.config.enable_prefix_caching,
                **self.coordinator_options,
            )
        address = [self.service.address if self.service else None]
        torch.distributed.broadcast_object_list(address, src=0)
        self.address = address[0]
        self._observe_forwards()
        await self.engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=int(self.address.rsplit(":", 1)[-1]),
            launch_inference_coordinator=False,
            hostname="127.0.0.1",
        )
        if self.rank == 0:
            await until(self.service.ready.is_set)
            assert len(self.service.coordinator.identities_of_data_parallel_ranks) == self.dp_size
            for _ in range(self.dp_size):
                client = InferenceClient(self.address)
                client.start(connect_timeout_seconds=30)
                self.clients.append(client)
        await self.barrier()

    def _observe_forwards(self):
        model = self.engine.controller.inference_wrapped_model.model
        forward = model.forward

        def observed_forward(*args, **kwargs):
            context = self.engine.context
            ids = [
                request_id
                for request_id in context.request_ids[
                    context.paused_request_count : context.total_request_count
                ].tolist()
                if request_id in self.engine.requests
            ]
            snapshot = dict(
                ids=ids,
                prefill=context.num_prefill_requests,
                decode=context.num_decode_requests,
                chunk=context.chunked_prefill_request_id,
                tokens=context.active_token_count,
                padded_tokens=context.padded_active_token_count,
                async_step=context.async_sched_step_count,
                cached={rid: self.engine.get_request(rid).num_cached_tokens for rid in ids},
            )
            result = forward(*args, **kwargs)
            # Commit evidence only after the actual model forward returns.
            if ids:
                self.witnesses.append(snapshot)
            return result

        model.forward = observed_forward

    async def pause(self):
        if self.rank == 0:
            self.clients[0].pause_engines()
        await asyncio.wait_for(self.engine.wait_until(EngineState.PAUSED), timeout=60)
        await self.barrier()

    async def unpause(self):
        if self.rank == 0:
            self.clients[0].unpause_engines()
        await asyncio.wait_for(self.engine.wait_until(EngineState.RUNNING), timeout=60)
        await self.barrier()

    async def run_paused_batch(self, submissions):
        """Return replies, actual owners and client-local IDs for an ordered batch."""
        await self.pause()
        payload = [None]
        if self.rank == 0:
            pending, local_ids = [], []
            coordinator = self.service.coordinator
            for client, prompt, params, *media in submissions:
                request_id, future = self.clients[client].add_request_with_id(
                    prompt, copy.deepcopy(params), multi_modal_data=media[0] if media else None
                )
                pending.append(future)
                local_ids.append((client, request_id))
                await until(lambda: len(coordinator.request_id_to_rank) == len(pending))
            owners = dict(coordinator.request_id_to_rank)
            assert all(
                coordinator._pending_counts[index] == list(owners.values()).count(identity)
                for identity, index in coordinator.identity_to_rank_index.items()
            )
        await self.unpause()
        if self.rank == 0:
            payload[0] = (await asyncio.wait_for(asyncio.gather(*pending), 120), owners, local_ids)
        await self.barrier()
        torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    def assert_retired(self):
        assert not self.engine.requests
        assert not self.engine.waiting_request_ids
        assert self.engine.context.get_active_request_count() == 0
        if self.rank == 0:
            self.service.assert_requests_retired()
            for client in self.clients:
                assert not client.completion_futures
                assert not client.request_submission_times
                assert not client.streams

    async def close(self):
        if hasattr(self.engine, "engine_loop_task"):
            await self.pause()
            if self.rank == 0:
                self.clients[0].stop_engines()
            await asyncio.wait_for(self.engine.engine_loop_task, timeout=60)
        await self.barrier()
        if self.rank == 0 and self.service is not None:
            if self.clients:
                self.clients[0].shutdown_coordinator()
            await asyncio.to_thread(self.service.thread.join, 30)
            assert not self.service.thread.is_alive()
            assert not self.service.errors, self.service.errors
            for client in self.clients:
                client.stop()
            await asyncio.gather(*(c.listener_task for c in self.clients), return_exceptions=True)
        self.sync.close()
        self.sync_context.term()


@asynccontextmanager
async def routed_model(monkeypatch, *, coordinator_options=None, engine_factory=None, **overrides):
    """Build a real distributed engine, optionally via a config-to-engine factory.

    Backend globals must be selected by the caller before entering this fixture.
    """
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    settings = dict(
        num_requests=0,
        max_sequence_length=256,
        context_max_requests=8,
        context_max_tokens=128,
        context_block_size_tokens=16,
        context_buffer_size_gb=0.02,
        transformer_impl="transformer_engine",
        flash_attention_version=3,
    )
    settings.update(overrides)
    config = engine_tests.DynamicEngineTestConfig(**settings)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        expert_model_parallel_size=config.expert_model_parallel_size,
    )
    harness = None
    try:
        with torch.inference_mode():
            harness = RoutedModel(config, engine_factory, **(coordinator_options or {}))
            yield harness
    finally:
        if harness is not None:
            await harness.close()
        delete_cuda_graphs()
        Utils.destroy_model_parallel()


def greedy_params(**overrides):
    params = dict(num_tokens_to_generate=8, termination_id=-1, top_k=1)
    params.update(overrides)
    return SamplingParams(**params)


def batch_invariant_engine(config):
    """Construct native-TE BI modules; caller pins backend and rounders before CUDA."""
    assert config.tensor_model_parallel_size == config.pipeline_model_parallel_size == 1
    torch.manual_seed(config.random_seed)
    engine_tests.model_parallel_cuda_manual_seed(
        config.random_seed, inference_rng_tracker=True, force_reset_rng=True
    )
    model_config = engine_tests.TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend="te_native",
        flash_attention_version=3,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.flash,
        transformer_impl="transformer_engine",
        nccl_all_reduce_for_prefill=False,
        inference_rng_tracker=True,
        inference_sampling_seed=config.random_seed,
    )
    model = (
        engine_tests.GPTModel(
            config=model_config,
            transformer_layer_spec=engine_tests.get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            position_embedding_type="rope",
        )
        .cuda()
        .eval()
    )
    context = engine_tests.DynamicInferenceEngineTestBase._build_inference_context(
        config, model_config, []
    )
    wrapper = engine_tests.GPTInferenceWrapper(Float16Module(model_config, model).eval(), context)
    controller = engine_tests.TextGenerationController(wrapper, DummyTokenizer(config.vocab_size))
    return engine_tests.DynamicInferenceEngine(controller, context)
