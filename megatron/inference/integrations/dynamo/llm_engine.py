# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shlex
import signal
import sys
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional

from dynamo._core import Context
from dynamo.common.backend.disagg import require_prefill_result
from dynamo.common.backend.engine import EngineConfig, GenerateChunk, GenerateRequest, LLMEngine
from dynamo.common.backend.health_check import build_health_check_payload, is_probe
from dynamo.common.backend.publisher import KvEventSource, PushSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import KvEventPublisher, ModelInput

from megatron.core.inference.inference_client import InferenceClient, InferenceRequestError
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference.integrations.dynamo.args import Config, parse_args
from megatron.inference.integrations.dynamo.telemetry import EngineEventReceiver

logger = logging.getLogger(__name__)


def build_sampling_params(request: GenerateRequest) -> SamplingParams:
    sampling = request.get("sampling_options") or {}
    stop = request.get("stop_conditions") or {}
    params = SamplingParams()
    if sampling.get("temperature") is not None:
        params.temperature = float(sampling["temperature"])
    if sampling.get("top_p") is not None:
        params.top_p = float(sampling["top_p"])
    if sampling.get("top_k") is not None:
        params.top_k = max(0, int(sampling["top_k"]))
    if stop.get("max_tokens") is not None:
        params.num_tokens_to_generate = int(stop["max_tokens"])
    if stop.get("stop"):
        params.stop_words = list(stop["stop"])
    stop_token_ids = list(stop.get("stop_token_ids") or [])
    if stop.get("ignore_eos"):
        params.termination_id = -1
    elif stop_token_ids:
        params.termination_id = int(stop_token_ids[0])
        if len(stop_token_ids) > 1:
            logger.warning(
                "Megatron supports one termination token; ignoring %d additional IDs",
                len(stop_token_ids) - 1,
            )
    output = request.get("output_options") or {}
    token_logprobs = output.get("logprobs")
    prompt_logprobs = output.get("prompt_logprobs")
    params.top_n_logprobs = max(
        int(token_logprobs or 0), int(prompt_logprobs or 0)
    )
    params.return_log_probs = token_logprobs is not None
    params.skip_prompt_log_probs = prompt_logprobs is None
    params.add_attributes({})
    if params.temperature == 0.0:
        params.top_k = 1
        params.top_p = 0.0
    return params


class MegatronLLMEngine(LLMEngine):
    """Unified Dynamo backend for one self-owned Megatron DP replica."""

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        self.client: Any = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._process_monitor: Optional[asyncio.Task] = None
        self._log_tasks: list[asyncio.Task] = []
        self._shutting_down = False
        self._metadata: dict[str, Any] = {}
        self._ready_messages: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._kv_queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._publisher: Optional[KvEventPublisher] = None
        self._publisher_lock = threading.Lock()
        self._event_receiver: Optional[EngineEventReceiver] = None
        self._release_clients: dict[str, Any] = {}
        self._request_ids: dict[str, int] = {}
        self.worker_id: Optional[int] = None

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple["MegatronLLMEngine", WorkerConfig]:
        config = parse_args(argv)
        return cls(config), cls.worker_config(config)

    @staticmethod
    def worker_config(config: Config) -> WorkerConfig:
        mode = {
            "aggregated": DisaggregationMode.AGGREGATED,
            "prefill": DisaggregationMode.PREFILL,
            "decode": DisaggregationMode.DECODE,
        }[config.role]
        return WorkerConfig(
            namespace=config.namespace,
            component=config.component,
            endpoint=config.endpoint,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
            disaggregation_mode=mode,
            enable_kv_routing=True,
        )

    async def start(self, worker_id: int) -> EngineConfig:
        self.worker_id = int(worker_id)
        if not os.path.isdir(self.config.megatron_root):
            raise FileNotFoundError(
                f"Megatron root does not exist: {self.config.megatron_root}"
            )

        self._event_receiver = EngineEventReceiver(
            self._on_engine_event,
            self.config.parent_event_host,
        )
        parent_event_address = self._event_receiver.start()
        command = self._engine_command(parent_event_address)
        logger.info("Launching owned Megatron engine: %s", " ".join(command))
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.config.megatron_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception:
            self._event_receiver.stop()
            self._event_receiver = None
            raise
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._log_tasks = [
            asyncio.create_task(self._forward_logs(self._process.stdout, logging.INFO)),
            asyncio.create_task(self._forward_logs(self._process.stderr, logging.WARNING)),
        ]

        readiness = await self._wait_for_readiness()
        self.client = InferenceClient(
            str(readiness["coordinator_address"]), deserialize=False
        )
        self.client.start(
            loop=asyncio.get_running_loop(),
            connect_timeout_seconds=min(30.0, self.config.engine_start_timeout),
        )
        self._metadata = dict(readiness["engine"])
        self._metadata["coordinator_address"] = str(readiness["coordinator_address"])
        self._process_monitor = asyncio.create_task(self._monitor_process())
        identity = {
            "worker_id": self.worker_id,
            "namespace": self.config.namespace,
            "component": self.config.component,
            "endpoint": self.config.endpoint,
            "role": self.config.role,
        }
        logger.info("Dynamo worker identity: %s", json.dumps(identity, sort_keys=True))
        if self.config.worker_id_file is not None:
            path = Path(self.config.worker_id_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(identity, sort_keys=True) + "\n")
            os.replace(temporary, path)

        return EngineConfig(
            model=self.config.model,
            served_model_name=self.config.served_model_name,
            context_length=int(self._metadata["context_length"]),
            kv_cache_block_size=int(self._metadata["kv_cache_block_size"]),
            total_kv_blocks=int(self._metadata["total_kv_blocks"]),
            max_num_seqs=int(self._metadata["max_num_seqs"]),
            max_num_batched_tokens=int(self._metadata["max_num_batched_tokens"]),
            data_parallel_size=1,
            data_parallel_start_rank=0,
            runtime_data={"role": self.config.role, "worker_id": self.worker_id},
        )

    def _engine_command(self, parent_event_address: str) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
        ]
        if self.config.launcher == "local":
            command.append("--standalone")
        else:
            command.extend(
                [
                    f"--nnodes={self.config.nnodes}",
                    f"--nproc-per-node={self.config.nproc_per_node}",
                    "--node-rank=__SLURM_NODE_RANK__",
                    f"--master-addr={self.config.master_addr}",
                    f"--master-port={self.config.master_port}",
                ]
            )
        if self.config.launcher == "local":
            command.append(f"--nproc-per-node={self.config.nproc_per_node}")
        command.extend(
            [
                "--module",
                "megatron.inference.integrations.dynamo.engine_service",
                "--dynamo-parent-event-address",
                parent_event_address,
                "--role",
                self.config.role,
            ]
        )
        if self.config.coordinator_host is not None:
            command.extend(
                ["--coordinator-host", self.config.coordinator_host]
            )
        if self.config.coordinator_port is not None:
            command.extend(
                ["--coordinator-port", str(self.config.coordinator_port)]
            )
        command.extend(self.config.megatron_argv)
        if self.config.launcher == "slurm":
            shell_command = shlex.join(command).replace(
                "__SLURM_NODE_RANK__", '"${SLURM_NODEID}"'
            )
            srun_command = [
                "srun",
                f"--nodes={self.config.nnodes}",
                f"--ntasks={self.config.nnodes}",
                "--ntasks-per-node=1",
                f"--gpus-per-node={self.config.nproc_per_node}",
                "--kill-on-bad-exit=1",
            ]
            if self.config.slurm_nodelist is not None:
                srun_command.append(f"--nodelist={self.config.slurm_nodelist}")
            return srun_command + ["bash", "-c", f"exec {shell_command}"]
        return command

    async def _forward_logs(
        self, stream: asyncio.StreamReader, level: int
    ) -> None:
        while line := await stream.readline():
            logger.log(level, "[megatron-engine] %s", line.decode(errors="replace").rstrip())

    async def _wait_for_readiness(self) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.config.engine_start_timeout
        while True:
            try:
                return self._ready_messages.get_nowait()
            except queue.Empty:
                pass
            if self._process is None:
                raise RuntimeError("Megatron process disappeared during startup")
            if self._process.returncode is not None:
                raise RuntimeError(
                    "Megatron engine exited before readiness "
                    f"with code {self._process.returncode}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the owned Megatron engine to become ready"
                )
            await asyncio.sleep(0.05)

    async def _monitor_process(self) -> None:
        assert self._process is not None
        returncode = await self._process.wait()
        if not self._shutting_down:
            logger.error("Owned Megatron engine exited unexpectedly: %d", returncode)
            os.kill(os.getpid(), signal.SIGTERM)

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self.client is None:
            raise RuntimeError("Megatron engine is not initialized")
        token_ids = list(request.get("token_ids") or [])
        if not token_ids:
            raise ValueError("Megatron backend requires token_ids")
        params = build_sampling_params(request)
        n = int((request.get("sampling_options") or {}).get("n") or 1)
        if n != 1:
            raise ValueError("Megatron Dynamo backend currently supports sampling n=1")
        context_id = str(context.id())

        probe = is_probe(request)
        if probe and self.config.role == "decode":
            yield {
                "token_ids": [],
                "index": 0,
                "finish_reason": "stop",
                "completion_usage": {
                    "prompt_tokens": len(token_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(token_ids),
                },
            }
            return
        if self.config.role == "prefill" and not probe:
            params.do_kv_handoff = True
            params.num_tokens_to_generate = 0
            stream = self.client.add_request_streaming(token_ids, params)
            self._request_ids[context_id] = stream.request_id
            final = None
            try:
                async for item in stream:
                    if "final" in item:
                        final = item["final"]
                        break
                if final is None:
                    raise RuntimeError("Megatron prefill stream ended without a result")
                disagg = dict(final.get("disaggregated_params") or {})
                disagg["release"] = {
                    "coordinator_addr": self._metadata["coordinator_address"],
                    "request_id": int(
                        disagg.get("request_id", final.get("request_id", stream.request_id))
                    ),
                }
                yield {
                    "token_ids": [],
                    "index": 0,
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": len(token_ids),
                        "completion_tokens": 0,
                        "total_tokens": len(token_ids),
                    },
                    "disaggregated_params": disagg,
                }
            finally:
                if final is None:
                    await stream.aclose()
                self._request_ids.pop(context_id, None)
            return

        release: dict[str, Any] = {}
        if self.config.role == "decode" and not probe:
            prefill = require_prefill_result(request, DisaggregationMode.DECODE)
            disagg = prefill.get("disaggregated_params") or {}
            release = disagg.get("release") or {}
            stream = self.client.add_request_with_kv_handoff_streaming(
                token_ids,
                params,
                disagg.get("kv_meta") or {},
                list(disagg.get("block_ids") or []),
            )
        else:
            stream = self.client.add_request_streaming(token_ids, params)
        self._request_ids[context_id] = stream.request_id

        released = False
        source_safe = False
        try:
            async for chunk in self._stream_chunks(stream, token_ids, params):
                source_safe = True
                yield chunk
                if not released and self.config.role == "decode":
                    released = await self._release_handoff_from_meta_async(release)
        except InferenceRequestError as error:
            source_safe = error.source_safe
            raise
        finally:
            self._request_ids.pop(context_id, None)
            if not released and self.config.role == "decode":
                try:
                    if not source_safe:
                        self.client.abort_request(stream.request_id)
                        source_safe = await self.client.wait_for_abort(stream.request_id)
                    if source_safe:
                        await self._release_handoff_from_meta_async(release)
                except Exception:
                    logger.exception("Failed to finish cancelled Megatron handoff")
                finally:
                    self.client.forget_abort(stream.request_id)

    async def _stream_chunks(
        self, stream, prompt_token_ids: list[int], params: SamplingParams
    ) -> AsyncGenerator[GenerateChunk, None]:
        completion_tokens = 0
        completed = False
        try:
            async for item in stream:
                if "partial" in item:
                    tokens = list(item["partial"].get("new_tokens") or [])
                    completion_tokens += len(tokens)
                    yield {"token_ids": tokens, "index": 0}
                    continue

                final = item.get("final")
                if final is None:
                    continue
                all_tokens = list(final.get("generated_tokens") or [])
                tokens = all_tokens[completion_tokens:]
                completion_tokens = len(all_tokens)
                max_tokens = params.num_tokens_to_generate
                chunk: GenerateChunk = {
                    "token_ids": tokens,
                    "index": 0,
                    "finish_reason": (
                        "length"
                        if max_tokens is not None and completion_tokens >= max_tokens
                        else "stop"
                    ),
                    "completion_usage": {
                        "prompt_tokens": len(prompt_token_ids),
                        "completion_tokens": completion_tokens,
                        "total_tokens": len(prompt_token_ids) + completion_tokens,
                    },
                }
                completed = True
                yield chunk
                return
            raise RuntimeError("Megatron stream ended without a terminal result")
        finally:
            if not completed:
                await stream.aclose()

    def _release_remote_handoff(self, address: str, request_id: int) -> None:
        client = self._release_clients.get(address)
        if client is None:
            client = InferenceClient(address, deserialize=False)
            client.start()
            self._release_clients[address] = client
        client.release_handoff(request_id)

    def _release_handoff_from_meta(self, release: dict[str, Any]) -> bool:
        address = release.get("coordinator_addr")
        request_id = release.get("request_id")
        if address is None or request_id is None:
            return False
        self._release_remote_handoff(str(address), int(request_id))
        return True

    async def _release_handoff_from_meta_async(self, release: dict[str, Any]) -> bool:
        """Release source state without blocking Dynamo's request loop."""

        if release.get("coordinator_addr") is None or release.get("request_id") is None:
            return False
        return await asyncio.to_thread(self._release_handoff_from_meta, release)

    async def abort(self, context: Context) -> None:
        request_id = self._request_ids.pop(str(context.id()), None)
        if request_id is not None and self.client is not None:
            self.client.abort_request(request_id)
            try:
                await self.client.wait_for_abort(request_id)
            finally:
                self.client.forget_abort(request_id)

    async def drain(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.config.drain_timeout
        while self._request_ids:
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("Timed out draining Megatron requests")
                return
            await asyncio.sleep(0.05)

    async def cleanup(self) -> None:
        self._shutting_down = True
        for client in self._release_clients.values():
            try:
                client.stop()
            except Exception:
                logger.exception("Failed to close Megatron handoff client")
        self._release_clients.clear()
        client = self.client
        self.client = None
        if client is not None:
            try:
                client.pause_engines()
                client.stop_engines()
                client.shutdown_coordinator()
            except Exception:
                logger.exception("Graceful Megatron coordinator shutdown failed")

        await self._wait_or_terminate_process()
        if client is not None:
            try:
                client.stop()
            except Exception:
                logger.exception("Failed to close Megatron coordinator client")
        if self._process_monitor is not None:
            self._process_monitor.cancel()
            await asyncio.gather(self._process_monitor, return_exceptions=True)
            self._process_monitor = None
        if self._event_receiver is not None:
            self._event_receiver.stop()
            self._event_receiver = None
        with self._publisher_lock:
            self._publisher = None
        if self._log_tasks:
            await asyncio.gather(*self._log_tasks, return_exceptions=True)
            self._log_tasks.clear()

    async def _wait_or_terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            await asyncio.wait_for(
                process.wait(), timeout=self.config.engine_shutdown_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Terminating unresponsive Megatron engine process group")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.error("Killing unresponsive Megatron engine process group")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        finally:
            self._process = None

    async def health_check_payload(self) -> dict[str, Any] | None:
        if self.client is None:
            return None
        return build_health_check_payload(int(self._metadata.get("bos_token_id", 0)))

    async def kv_event_sources(self) -> list[KvEventSource]:
        if self.config.role == "decode" or self.client is None:
            return []
        if not self._metadata.get("enable_prefix_caching", False):
            return []
        return [PushSource(on_ready=self._set_publisher, dp_rank=0)]

    def _on_engine_event(self, kind: str, payload: dict) -> None:
        if kind == "ready":
            try:
                self._ready_messages.put_nowait(payload)
            except queue.Full:
                logger.warning("Ignoring duplicate Megatron readiness message")
            return
        if kind in ("stored", "removed", "cleared"):
            with self._publisher_lock:
                if self._publisher is None:
                    self._kv_queue.put((kind, payload))
                else:
                    self._publish_event(self._publisher, kind, payload)
            return
        logger.warning("Ignoring unknown Megatron engine event %s", kind)

    def _set_publisher(self, publisher: KvEventPublisher) -> None:
        with self._publisher_lock:
            self._publisher = publisher
            try:
                while True:
                    kind, payload = self._kv_queue.get_nowait()
                    self._publish_event(publisher, kind, payload)
            except queue.Empty:
                pass

    @staticmethod
    def _publish_event(publisher: KvEventPublisher, kind: str, payload: dict) -> None:
        try:
            if kind == "stored":
                publisher.publish_stored(**payload)
            elif kind == "removed":
                publisher.publish_removed(**payload)
            else:
                publisher.publish_all_cleared()
        except Exception:
            logger.exception("Failed to publish Megatron KV event %s", kind)
