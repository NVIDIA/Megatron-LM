# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Diffusion inference engine for two-tower models."""

import itertools
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F

from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams

if TYPE_CHECKING:
    from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizer


class DiffusionEngine(AbstractEngine):
    """Inference engine wrapping :class:`TwoTowerMambaModel`.

    Provides an API compatible with :class:`MegatronServer` and
    :func:`run_mcore_engine`.  Three modes:

    * **AR generation** (``tokens_to_generate > 0`` and
      ``_single_tower_mode``): auto-regressive decoding via
      :meth:`TwoTowerMambaModel.generate_ar`.
    * **Diffusion generation** (``tokens_to_generate > 0`` and
      not ``_single_tower_mode``): block-wise mask-diffusion via
      :meth:`TwoTowerMambaModel.generate_diffusion`.
    * **Loglikelihood** (``tokens_to_generate == 0``): context-tower
      forward for next-token log-probabilities via
      :meth:`TwoTowerMambaModel.forward_for_likelihood`.

    Args:
        model (torch.nn.Module): :class:`TwoTowerMambaModel` instance
            (possibly wrapped by DDP).
        tokenizer (MegatronTokenizer): Tokenizer with ``tokenize`` / ``detokenize`` methods.
        block_size (int): Tokens per diffusion block.
        steps_per_block (int): Denoising iterations per block.
        sampling_strategy (str): ``"predict_and_noise"``, ``"posterior"``,
            or ``"confidence_unmasking"``.
        posterior_float64 (bool): Use float64 for posterior computation.
        noise_schedule (str): Noise schedule type (``"linear"``, etc.).
        confidence_threshold (float): Threshold for confidence unmasking.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: 'MegatronTokenizer',
        block_size: int = 64,
        steps_per_block: int = 1,
        sampling_strategy: str = "predict_and_noise",
        posterior_float64: bool = False,
        noise_schedule: str = "linear",
        confidence_threshold: float = 1e6,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.steps_per_block = steps_per_block
        self.sampling_strategy = sampling_strategy
        self.posterior_float64 = posterior_float64
        self.noise_schedule = noise_schedule
        self.confidence_threshold = confidence_threshold
        self._request_counter = itertools.count()

        self._inner_model = model
        if hasattr(self._inner_model, 'module'):
            self._inner_model = self._inner_model.module

    @property
    def controller(self) -> 'DiffusionEngine':
        """Compatibility property for :func:`run_mcore_engine`."""
        if not hasattr(self, "sampling_rng"):
            self.sampling_rng = torch.Generator(device=torch.cuda.current_device())
            self.sampling_rng.manual_seed(0)
        return self

    def get_new_request_id(self) -> int:
        """Get a new unique request ID."""
        return next(self._request_counter)

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        sampling_params: Optional[SamplingParams] = None,
        inference_requests: Optional[List[InferenceRequest]] = None,
        **kwargs,
    ) -> List[InferenceRequest]:
        """Generate completions or compute loglikelihoods.

        Args:
            prompts (Optional[List[str]]): List of prompt strings (alternative
                to *inference_requests*).
            sampling_params (Optional[SamplingParams]): Sampling parameters.
            inference_requests (Optional[List[InferenceRequest]]): Pre-constructed
                inference requests.

        Returns:
            List[InferenceRequest]: Completed requests with generated text
            and / or log probabilities filled in.
        """
        if inference_requests is None:
            if prompts is None:
                raise ValueError("Either prompts or inference_requests must be provided")
            inference_requests = []
            for prompt in prompts:
                prompt_tokens = self.tokenizer.tokenize(prompt)
                req = InferenceRequest(
                    request_id=self.get_new_request_id(),
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    sampling_params=sampling_params or SamplingParams(),
                )
                inference_requests.append(req)

        tokens_to_generate = inference_requests[0].sampling_params.num_tokens_to_generate

        if tokens_to_generate == 0:
            results = self._compute_loglikelihood(inference_requests)
        else:
            results = self._generate(inference_requests)

        for req in results:
            req.prompt_top_n_logprobs = []
            req.generated_top_n_logprobs = []

        return results

    def _compute_loglikelihood(
        self, inference_requests: List[InferenceRequest]
    ) -> List[InferenceRequest]:
        """Compute log probabilities for input tokens via the context tower.

        Uses varlen-packed prefill (no padding) with the same token budget
        as ``_prefill`` to keep memory bounded.

        Args:
            inference_requests (List[InferenceRequest]): Requests whose
                ``prompt_tokens`` will be scored.

        Returns:
            List[InferenceRequest]: Requests with ``prompt_log_probs`` filled in
            and empty ``generated_text``.
        """
        prompt_tokens_list = []
        for req in inference_requests:
            pt = req.prompt_tokens
            if isinstance(pt, list):
                pt = torch.tensor(pt, dtype=torch.long, device='cuda')
            else:
                pt = pt.to('cuda')
            prompt_tokens_list.append(pt)

        with torch.no_grad():
            per_request_logits = self._inner_model.forward_for_likelihood(prompt_tokens_list)

        results = []
        for i, req in enumerate(inference_requests):
            logits_i = per_request_logits[i]  # (seq_len, V)
            ids_i = prompt_tokens_list[i]  # (seq_len,)

            shift_logits = logits_i[:-1, :]
            shift_labels = ids_i[1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            real_log_probs = torch.cat([torch.zeros(1, device=token_lp.device), token_lp]).tolist()

            result = InferenceRequest(
                request_id=req.request_id,
                prompt=req.prompt,
                prompt_tokens=(
                    req.prompt_tokens
                    if isinstance(req.prompt_tokens, list)
                    else req.prompt_tokens.tolist()
                ),
                sampling_params=req.sampling_params,
                generated_text="",
                generated_tokens=torch.tensor([], dtype=torch.long),
                prompt_log_probs=real_log_probs,
                generated_log_probs=[],
            )
            results.append(result)
        return results

    def _generate(self, inference_requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Generate text via AR or diffusion depending on ``_single_tower_mode``.

        Args:
            inference_requests (List[InferenceRequest]): Requests to generate
                completions for.

        Returns:
            List[InferenceRequest]: Requests with ``generated_text`` and
            ``generated_tokens`` filled in.
        """
        prompt_tokens_list = []
        for req in inference_requests:
            pt = req.prompt_tokens
            if isinstance(pt, list):
                pt = torch.tensor(pt, dtype=torch.long)
            prompt_tokens_list.append(pt.cuda())

        sp = inference_requests[0].sampling_params
        max_new_tokens = sp.num_tokens_to_generate

        with torch.no_grad():
            if getattr(self._inner_model, '_single_tower_mode', False):
                output_list, nfe = self._inner_model.generate_ar(
                    prompt_ids_list=prompt_tokens_list,
                    max_new_tokens=max_new_tokens,
                    temperature=sp.temperature,
                    top_k=sp.top_k if sp.top_k > 0 else None,
                    top_p=sp.top_p if sp.top_p > 0 else None,
                )
            else:
                if max_new_tokens % self.block_size != 0:
                    max_new_tokens = ((max_new_tokens // self.block_size) + 1) * self.block_size

                output_list, nfe = self._inner_model.generate_diffusion(
                    prompt_ids_list=prompt_tokens_list,
                    max_new_tokens=max_new_tokens,
                    block_length=self.block_size,
                    steps_per_block=self.steps_per_block,
                    temperature=sp.temperature,
                    top_k=sp.top_k if sp.top_k > 0 else None,
                    top_p=sp.top_p if sp.top_p > 0 else None,
                    sampling_strategy=self.sampling_strategy,
                    posterior_float64=self.posterior_float64,
                    noise_schedule=self.noise_schedule,
                    confidence_threshold=self.confidence_threshold,
                )

        results = []
        for i, req in enumerate(inference_requests):
            prompt_len = prompt_tokens_list[i].shape[0]
            generated_tokens = output_list[i][prompt_len:].cpu()
            generated_tokens = generated_tokens[: sp.num_tokens_to_generate]

            generated_text = self.tokenizer.detokenize(generated_tokens.tolist())

            result = InferenceRequest(
                request_id=req.request_id,
                prompt=req.prompt,
                prompt_tokens=(
                    req.prompt_tokens
                    if isinstance(req.prompt_tokens, list)
                    else req.prompt_tokens.tolist()
                ),
                sampling_params=req.sampling_params,
                generated_text=generated_text,
                generated_tokens=generated_tokens,
                prompt_log_probs=[],
                generated_log_probs=[],
                segments=[req.prompt, generated_text],
            )
            results.append(result)
        return results
