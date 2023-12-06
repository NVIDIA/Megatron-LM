import contextlib
import itertools
from typing import Iterator, List, Union

import torch

from megatron import core, get_args, get_num_microbatches, print_rank_0, get_tokenizer
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import auto_schedule, v_schedule
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)
from megatron.core.weight_grad_store import WeightGradStore
from megatron.timers import Timer
from megatron.utils import is_second_last_pipeline_stage


AUTO_SCHEDULE_COMMUNICATION_TYPES = {'RECV_FORWARD', 'RECV_BACKWARD', 'SEND_FORWARD', 'SEND_BACKWARD'}


class ScheduleTimers:
    f = Timer('f')
    b = Timer('b')
    w = Timer('w')
    f_cnt = 0
    b_cnt = 0
    w_cnt = 0
    f_mem = 0
    b_mem = 0
    w_mem = 0
    iter_counter = 0
    comm_time = 0
    concluded = False

    @classmethod
    def conclusion(cls):
        assert cls.concluded
        assert cls.f_cnt > 0
        assert cls.b_cnt > 0
        avg_f = cls.f.elapsed(reset=False) / cls.f_cnt * 1000
        avg_b = cls.b.elapsed(reset=False) / cls.b_cnt * 1000
        avg_f_mem = cls.f_mem / cls.f_cnt // 1000000
        avg_b_mem = cls.b_mem / cls.b_cnt // 1000000
        if cls.w_cnt > 0:
            avg_w = cls.w.elapsed(reset=False) / cls.w_cnt * 1000
        else:
            avg_w = avg_b
        avg_w_mem = 0 - avg_f_mem - avg_b_mem
        return (avg_f, avg_b, avg_w, cls.comm_time * 1000, 
            avg_f_mem, avg_b_mem, avg_w_mem)


def bootstrap_and_profile_p2p_communication(
    config, send_tensor_shapes, recv_tensor_shapes
    ):
    if ScheduleTimers.iter_counter == 1 and parallel_state.get_pipeline_model_parallel_world_size() > 1:
        nccl_init_tensor = [torch.Tensor([0]).cuda()]
        shape = [(1,)]
        if get_args().zero_bubble_v_schedule:
            # Make everyone think they are the first chunk, so we still need additional check to prevent rank -1 to send_forward/recv_backward
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_forward(shape, config)
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            send_forward(nccl_init_tensor, shape, config)
            recv_backward(shape, config)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            send_backward(nccl_init_tensor, shape, config)

        # Benchmarking the communication cost
        send_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in send_tensor_shapes]
        recv_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in recv_tensor_shapes]
        torch.distributed.barrier()
        t = Timer('comm-benchmark')
        t.start()
        print_rank_0(
            f"Start benchmarking communication with size {recv_tensor_shapes}, {send_tensor_shapes}")
        for i in range(10):
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_forward(recv_tensor_shapes, config)
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                send_forward(send_data, send_tensor_shapes, config)
                recv_backward(send_tensor_shapes, config)
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                send_backward(recv_data, recv_tensor_shapes, config)
        t.stop()
        per_communication = torch.cuda.FloatTensor([t.elapsed() / (
                parallel_state.get_pipeline_model_parallel_world_size() - 1) / 2 / 10])
        torch.distributed.all_reduce(per_communication,
                                     torch.distributed.ReduceOp.MAX)
        ScheduleTimers.comm_time = per_communication.item()
        print_rank_0(f"Communication time: {ScheduleTimers.comm_time}")


def fused_pipeline_ops(
    tensor_send_prev: List[torch.Tensor],
    tensor_recv_prev: List[torch.Tensor],
    tensor_send_next: List[torch.Tensor],
    tensor_recv_next: List[torch.Tensor],
):
    ops = []
    group = get_pipeline_model_parallel_group()
    for t in tensor_send_prev:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    for t in tensor_recv_prev:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    for t in tensor_send_next:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    for t in tensor_recv_next:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


class ZeroBubbleVPipeScheduler:

    def __init__(self):
        self._reset()
        self.tensor_shape = None
        self.config = None
        self.run_timer = None
        self.forward_step_func = None
        self.model = None
        self.data_iterator = None
        self.num_microbatches = None
        self.collect_non_loss_data = None
        self.forward_only = None
        self.model_type = None
        self.no_sync_context = None
        self.no_sync_func = None

        self.schedules = None
        self.it = 0
        self.do_post_validation = False
        self.is_first_run = True
        self.optimizer = None

    def _reset(self):
        # two dim array, first dim is the model chunk, second dim is the microbatch queue
        self.input_tensors = [[], []]
        self.output_tensors = [[], []]
        self.send_forward_buffer = [[], []]
        self.recv_forward_buffer = [[], []]
        self.send_backward_buffer = [[], []]
        self.recv_backward_buffer = [[], []]
        self.forward_data_store = []
        self.send_handles = []
        self.local_send_forward_buffer = []
        self.local_send_backward_buffer = []
        self.w_clear_run = [False, False]
        # map of {direction -> {node, shape}}
        self.communication_batch = {
            'SEND_NEXT': [],
            'RECV_NEXT': [],
            'SEND_PREV': [],
            'RECV_PREV': [],
        }


    @classmethod
    def direction_map(cls, node):
        if node.chunk == 0:
            return {
                'SEND_FORWARD': 'SEND_NEXT',
                'RECV_FORWARD': 'RECV_PREV',
                'SEND_BACKWARD': 'SEND_PREV',
                'RECV_BACKWARD': 'RECV_NEXT',
            }[node.type]
        else:
            return {
                'SEND_FORWARD': 'SEND_PREV',
                'RECV_FORWARD': 'RECV_NEXT',
                'SEND_BACKWARD': 'SEND_NEXT',
                'RECV_BACKWARD': 'RECV_PREV',
            }[node.type]

    def buffer_map(self, node):
        return {
            'SEND_FORWARD': self.send_forward_buffer[node.chunk],
            'RECV_FORWARD': self.recv_forward_buffer[node.chunk],
            'SEND_BACKWARD': self.send_backward_buffer[node.chunk],
            'RECV_BACKWARD': self.recv_backward_buffer[node.chunk],
        }[node.type]

    def flush(self):
        name = '_'.join(
            [f'{v[0].type}.{v[0].chunk}.{v[0].minibatch}' for v in itertools.chain(*[vs for k, vs in self.communication_batch.items()])])
        sn_tensors = [
            self.buffer_map(x[0]).pop(0)
            for x in self.communication_batch['SEND_NEXT']
        ]
        sp_tensors = [
            self.buffer_map(x[0]).pop(0)
            for x in self.communication_batch['SEND_PREV']
        ]
        rn_tensors = [
            torch.empty(
                self.tensor_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_NEXT']
        ]
        rp_tensors = [
            torch.empty(
                self.tensor_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_PREV']
        ]
        if get_args().profile:
            torch.cuda.nvtx.range_push(name)
        req = fused_pipeline_ops(
            sp_tensors,
            rp_tensors,
            sn_tensors,
            rn_tensors
        )
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        # We don't care about the reqs order here, all users need to all reqs to finish
        for x in self.communication_batch['RECV_NEXT']:
            self.buffer_map(x[0]).append((rn_tensors.pop(0), req))
        for x in self.communication_batch['RECV_PREV']:
            self.buffer_map(x[0]).append((rp_tensors.pop(0), req))
        self.send_handles.append(req)
        assert(not rn_tensors)
        assert(not rp_tensors)
        for direction in ['SEND_PREV', 'SEND_NEXT']:
            for id, x in enumerate(self.communication_batch[direction]):
                if x[0].type == 'SEND_FORWARD':
                    deallocate_output_tensor(sp_tensors[id] if direction == 'SEND_PREV' else sn_tensors[id],
                                             self.config.deallocate_pipeline_outputs)
        for k, v in self.communication_batch.items():
            v.clear()


    def add_communication(
        self,
        scheduled_node: auto_schedule.ScheduledNode,
        next_is_comm: bool,
        next_compute: auto_schedule.ScheduledNode
    ):
        if self.forward_only and 'BACKWARD' in scheduled_node.type:
            return
        self.communication_batch[self.direction_map(scheduled_node)].append(
            (scheduled_node, self.tensor_shape))
        def is_consumer(scheduled_node, next_compute):
            if scheduled_node.chunk == next_compute.chunk and scheduled_node.minibatch == next_compute.minibatch:
                if scheduled_node.type == 'RECV_FORWARD' and next_compute.type == 'F':
                    return True
                if scheduled_node.type == 'RECV_BACKWARD' and next_compute.type == 'B':
                    return True
            return False
        if (next_compute is not None and is_consumer(scheduled_node, next_compute)) or not next_is_comm or self.forward_only:
            self.flush()

    def schedule_f(self, scheduled_node):
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = None
        elif scheduled_node.chunk == 1 and core.parallel_state.is_pipeline_last_stage(
            ignore_virtual=True):
            input_tensor = self.local_send_forward_buffer.pop(0)
        else:
            input_tensor = self.recv_forward_buffer[scheduled_node.chunk].pop(0)
            for h in input_tensor[1]:
                h.wait()
            input_tensor = input_tensor[0]
        if get_args().profile:
            torch.cuda.nvtx.range_push(
                f'F{scheduled_node.minibatch}.{scheduled_node.chunk}')
        if self.run_timer:
            ScheduleTimers.f_cnt += 1
            ScheduleTimers.f.start()
        output_tensor = forward_step(
            self.forward_step_func,
            self.data_iterator[scheduled_node.chunk],
            self.model[scheduled_node.chunk],
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.config,
            self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
        )
        if self.run_timer:
            ScheduleTimers.f.stop()
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        if not core.parallel_state.is_pipeline_last_stage():
            if scheduled_node.chunk == 0 and core.parallel_state.is_pipeline_last_stage(
                ignore_virtual=True):
                detached_output_tensor = output_tensor.detach()
                detached_output_tensor.requires_grad_()
                self.local_send_forward_buffer.append(detached_output_tensor)
                deallocate_output_tensor(output_tensor,
                                         self.config.deallocate_pipeline_outputs)
            else:
                self.send_forward_buffer[scheduled_node.chunk].append(output_tensor)
        if not self.forward_only:
            self.input_tensors[scheduled_node.chunk].append(input_tensor)
            self.output_tensors[scheduled_node.chunk].append(output_tensor)

    def schedule_b(self, scheduled_node):
        if not self.forward_only:
            input_tensor = self.input_tensors[scheduled_node.chunk].pop(0)
            output_tensor = self.output_tensors[scheduled_node.chunk].pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = None
            elif scheduled_node.chunk == 0 and core.parallel_state.is_pipeline_last_stage(
                ignore_virtual=True):
                output_tensor_grad = self.local_send_backward_buffer.pop(0)
            else:
                output_tensor_grad = self.recv_backward_buffer[
                    scheduled_node.chunk].pop(0)
                for h in output_tensor_grad[1]:
                    h.wait()
                output_tensor_grad = output_tensor_grad[0]
            if get_args().profile:
                torch.cuda.nvtx.range_push(
                    f'B{scheduled_node.minibatch}.{scheduled_node.chunk}')
            if self.run_timer:
                ScheduleTimers.b_cnt += 1
                ScheduleTimers.b.start()
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, self.model_type,
                self.config
            )
            if self.run_timer:
                ScheduleTimers.b.stop()
            if get_args().profile:
                torch.cuda.nvtx.range_pop()
            if not core.parallel_state.is_pipeline_first_stage():
                if scheduled_node.chunk == 1 and core.parallel_state.is_pipeline_last_stage(
                    ignore_virtual=True):
                    self.local_send_backward_buffer.append(input_tensor_grad)
                else:
                    self.send_backward_buffer[scheduled_node.chunk].append(
                        input_tensor_grad)
            WeightGradStore.flush(chunk=scheduled_node.chunk)

    def schedule_w(self, scheduled_node, non_w_pending):
        if not self.forward_only:
            chunk = scheduled_node.chunk

            if non_w_pending and scheduled_node.minibatch != self.num_microbatches - 1:
                if get_args().profile:
                    torch.cuda.nvtx.range_push(f'W{scheduled_node.minibatch}.{scheduled_node.chunk}')
                if self.run_timer:
                    ScheduleTimers.w_cnt += 1
                    ScheduleTimers.w.start()
                WeightGradStore.pop(chunk=scheduled_node.chunk)
                if self.run_timer:
                    ScheduleTimers.w.stop()
                if get_args().profile:
                    torch.cuda.nvtx.range_pop()
            elif not self.w_clear_run[chunk]:
                # Clear if this is the last minibatch or there is no non-W pending
                if get_args().profile:
                    torch.cuda.nvtx.range_push(f'W_clear.{chunk}')
                WeightGradStore.clear(self.model[chunk], chunk=chunk)
                if get_args().profile:
                    torch.cuda.nvtx.range_pop()  # W
                self.w_clear_run[chunk] = True

    def run_until_post_validation(self):
        optimizer = self.optimizer
        updated, grad_norm, rollback, succeed = None, None, None, None
        it = 0
        if optimizer.do_this_step:
            assert optimizer.do_prev_step
            for data_iter in self.data_iterator:
                if data_iter is None:
                    continue
                data_iter.clear_buffer()
                data_iter.save_to_buffer()
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                # print(f"True {rank}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD"]:
                    assert scheduled_node.chunk == 0
                    next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                    next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                    next_compute = next_compute[0] if len(next_compute) > 0 else None
                    self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == 'F':
                    assert scheduled_node.chunk == 0
                    self.schedule_f(scheduled_node)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation()
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert succeed is not None
        else:
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                # print(f"False {rank}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD", "F"]:
                    if optimizer.do_prev_step and scheduled_node.type == "RECV_FORWARD":
                        next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                        next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                        next_compute = next_compute[0] if len(next_compute) > 0 else None
                        self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation()
                    # print(f"{rank} False post validation done")
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert not succeed
        # print(f"{rank}: {optimizer.do_prev_step}, {optimizer.do_this_step} -> {succeed}")
        if not succeed:
            if optimizer.do_prev_step:
                # send dummy recv_forward to clear send_forward request of last rank
                while it < len(self.schedules):
                    scheduled_node = self.schedules[it]
                    parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                    if scheduled_node.type == "RECV_FORWARD" and scheduled_node.rollback:
                        # print(f"rollback {rank}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
                        self.add_communication(scheduled_node, False, None)
                    it += 1
            self._reset()
            it = 0
        for data_iter in self.data_iterator:
            if data_iter is None:
                continue
            if succeed:
                data_iter.clear_buffer()
            data_iter.pop_from_buffer()
        self.it = it
        return updated, grad_norm, rollback

    def run(self):
        self.disable_grad_sync()

        if get_args().profile:
            torch.cuda.nvtx.range_push(f'iter_{torch.distributed.get_rank()}_{ScheduleTimers.iter_counter}')

        it = self.it
        while it < len(self.schedules):
            scheduled_node = self.schedules[it]
            parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
            # print(f"iter {rank}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
            # print(f"rank {torch.distributed.get_rank()} {scheduled_node.type} CHUNK={scheduled_node.chunk} MB={scheduled_node.minibatch} ST={scheduled_node.start_time}")
            if "POST_VALIDATION" in scheduled_node.type:
                pass
            elif scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                next_compute = next_compute[0] if len(next_compute) > 0 else None
                self.add_communication(scheduled_node, next_is_comm, next_compute)
            elif scheduled_node.type == 'F':
                self.schedule_f(scheduled_node)
            elif scheduled_node.type == 'B':
                self.schedule_b(scheduled_node)
            elif scheduled_node.type == 'W':
                non_w_pending = any([node.type != 'W' for node in self.schedules[it + 1:]])
                self.schedule_w(scheduled_node, non_w_pending)
            else:
                raise ValueError(f"Unknown node type {scheduled_node.type}")
            it += 1
        self.it = it
        for h in self.send_handles:
            for hh in h:
                hh.wait()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()  # iter

        if not self.forward_only:
            # Launch any remaining grad reductions
            if self.no_sync_context is not None:
                self.enable_grad_sync()

            if self.config.finalize_model_grads_func is not None:
                # Finalize model grads (perform full grad all-reduce / reduce-scatter for
                # data parallelism, layernorm all-reduce for sequence parallelism).
                self.config.finalize_model_grads_func(self.model)

            if get_args().zero_bubble_pipeline_timers_end_iter == ScheduleTimers.iter_counter:
                ScheduleTimers.concluded = True

        return self.forward_data_store

    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            self.no_sync_context = self.no_sync_func()
            self.no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None

    def prepare(
        self,
        schedule: List[auto_schedule.ScheduledNode],
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ):
        assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
        assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
        assert isinstance(
            data_iterator, list
        ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
        config = get_model_config(model[0])
        self.config = config
        if config.overlap_p2p_comm and config.batch_p2p_comm:
            raise ValueError(
                "Can not use both overlap_p2p_comm and batch_p2p_comm")

        # Disable async grad reductions
        no_sync_func = config.no_sync_func
        if isinstance(no_sync_func, list):

            def multi_no_sync():
                stack = contextlib.ExitStack()
                for model_chunk_no_sync_func in config.no_sync_func:
                    stack.enter_context(model_chunk_no_sync_func())
                return stack

            no_sync_func = multi_no_sync
        assert no_sync_func is None, "Sync func is not supported yet"
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext
        self.no_sync_func = no_sync_func
        self.no_sync_context = None

        assert config.param_sync_func is None, "Param sync func is not supported yet"

        # Checkpoint the activations of partial Transformer layers in a number of micro-batches
        # within the maximum outstanding micro-batch backpropagations.
        # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
        # checkpoint partial Transformer layers (or skip checkpointing) and
        # the rest of micro-batches within a window of micro-batches checkpoint
        # all Transformer layers. The window of micro-batches is set by the maximum
        # outstanding backpropagations and becomes smaller at later pipeline stages.
        # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
        max_outstanding_backprops = None
        assert config.num_microbatches_with_partial_activation_checkpoints is None

        model_type = get_model_type(model[0])
        self.model_type = model_type

        tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
        self.tensor_shape = tensor_shape
        if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
            raise RuntimeError(
                "Interleaving is not supported with a different decoder sequence length."
            )

        rank = parallel_state.get_pipeline_model_parallel_rank()
        assert get_tensor_shapes(
            rank=rank - 1,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )[0] == tensor_shape
        assert get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )[0] == tensor_shape
        if not forward_only:
            ScheduleTimers.iter_counter += 1
        run_timer = (
            get_args().zero_bubble_pipeline_timers_end_iter
            >= ScheduleTimers.iter_counter
            >= get_args().zero_bubble_pipeline_timers_start_iter
        )
        bootstrap_and_profile_p2p_communication(config, [tensor_shape], [tensor_shape])
        self.run_timer = run_timer

        self.schedules = schedule
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.forward_only = forward_only
        self.collect_non_loss_data = collect_non_loss_data

        self._reset()
        self.it = 0

    def __call__(self, *args, **kwargs):
        if kwargs['forward_only']:
            self.prepare(*args, **kwargs)
            assert self.do_post_validation
            self.do_post_validation = True
            self.is_first_run = True
            return self.run()
        if not get_args().enable_optimizer_post_validation:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
            return self.run()
        # enable_optimizer_post_validation == True
        if self.is_first_run:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
        if self.do_post_validation:
            self.prepare(*args, **kwargs)
            result = self.run_until_post_validation()
            self.do_post_validation = False
        else:
            result = self.run()
            self.do_post_validation = True
        return result


class ZeroBubbleScheduler:

    def __init__(self):
        self._reset()

        self.schedules = None
        self.send_tensor_shapes = None
        self.recv_tensor_shapes = None
        self.config = None
        self.run_timer = None
        self.forward_step_func = None
        self.data_iterator = None
        self.model = None
        self.model_type = None
        self.num_microbatches = None
        self.collect_non_loss_data = None
        self.forward_only = None
        self.no_sync_context = None
        self.no_sync_func = None

        self.it = 0
        self.do_post_validation = False
        self.is_first_run = True
        self.optimizer = None

    def _reset(self):
        # Input, output tensors only need to be saved when doing backward passes
        self.input_tensors = []
        self.output_tensors = []
        self.send_forward_buffer = []
        self.recv_forward_buffer = []
        self.send_backward_buffer = []
        self.recv_backward_buffer = []
        self.forward_data_store = []
        self.send_handles = []
        self.communication_batch = {
            'SEND_NEXT': [],
            'RECV_NEXT': [],
            'SEND_PREV': [],
            'RECV_PREV': [],
        }

    @classmethod
    def direction_map(cls, node):
        return {
            'SEND_FORWARD': 'SEND_NEXT',
            'RECV_FORWARD': 'RECV_PREV',
            'SEND_BACKWARD': 'SEND_PREV',
            'RECV_BACKWARD': 'RECV_NEXT',
        }[node.type]

    def buffer_map(self, node):
        return {
            'SEND_FORWARD': self.send_forward_buffer,
            'RECV_FORWARD': self.recv_forward_buffer,
            'SEND_BACKWARD': self.send_backward_buffer,
            'RECV_BACKWARD': self.recv_backward_buffer,
        }[node.type]

    def flush(self):
        name = '_'.join(
            [f'{v[0].type}.{v[0].minibatch}' for v in itertools.chain(
                *[vs for k, vs in self.communication_batch.items()])])
        assert self.send_tensor_shapes == self.recv_tensor_shapes
        assert len(self.send_tensor_shapes) == 1
        sn_tensors = [
            self.buffer_map(x[0]).pop(0)[0]
            for x in self.communication_batch['SEND_NEXT']
        ]
        sp_tensors = [
            self.buffer_map(x[0]).pop(0)[0]
            for x in self.communication_batch['SEND_PREV']
        ]

        rn_tensors = [
            torch.empty(
                self.send_tensor_shapes[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_NEXT']
        ]
        rp_tensors = [
            torch.empty(
                self.send_tensor_shapes[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_PREV']
        ]
        if get_args().profile:
            torch.cuda.nvtx.range_push(name)
        req = fused_pipeline_ops(
            sp_tensors,
            rp_tensors,
            sn_tensors,
            rn_tensors
        )
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        # We don't care about the reqs order here, all users need to all reqs to finish
        for x in self.communication_batch['RECV_NEXT']:
            self.buffer_map(x[0]).append(([rn_tensors.pop(0)], [req]))
        for x in self.communication_batch['RECV_PREV']:
            self.buffer_map(x[0]).append(([rp_tensors.pop(0)], [req]))
        self.send_handles.append([req])
        assert(not rn_tensors)
        assert(not rp_tensors)
        for direction in ['SEND_PREV', 'SEND_NEXT']:
            for id, x in enumerate(self.communication_batch[direction]):
                if x[0].type == 'SEND_FORWARD':
                    deallocate_output_tensor(sp_tensors[id] if direction == 'SEND_PREV' else sn_tensors[id],
                                             self.config.deallocate_pipeline_outputs)
        for k, v in self.communication_batch.items():
            v.clear()

    def add_communication(
        self,
        scheduled_node: auto_schedule.ScheduledNode,
        next_is_comm: bool,
        next_compute: auto_schedule.ScheduledNode
    ):
        if self.forward_only and 'BACKWARD' in scheduled_node.type:
            return
        self.communication_batch[self.direction_map(scheduled_node)].append(
            (scheduled_node, None))
        def is_consumer(scheduled_node, next_compute):
            if scheduled_node.minibatch == next_compute.minibatch:
                if scheduled_node.type == 'RECV_FORWARD' and next_compute.type == 'F':
                    return True
                if scheduled_node.type == 'RECV_BACKWARD' and next_compute.type == 'B':
                    return True
            return False
        if (next_compute is not None and is_consumer(scheduled_node, next_compute)) or not next_is_comm or self.forward_only:
            self.flush()

    def schedule_f(self, scheduled_node):
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = [None] * len(self.recv_tensor_shapes)
        else:
            input_tensor = self.recv_forward_buffer.pop(0)
            for h in input_tensor[1]:
                for hh in h:
                    hh.wait()
            input_tensor = input_tensor[0]
        if get_args().profile:
            torch.cuda.nvtx.range_push(f'F{scheduled_node.minibatch}')
        if self.run_timer:
            ScheduleTimers.f_cnt += 1
            ScheduleTimers.f.start()
            mem_before = torch.cuda.memory_allocated()
        output_tensor = forward_step(
            self.forward_step_func,
            self.data_iterator,
            self.model,
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.config,
            self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
        )
        if self.run_timer:
            ScheduleTimers.f.stop()
            ScheduleTimers.f_mem += torch.cuda.memory_allocated() - mem_before
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        self.send_forward_buffer.append(output_tensor)
        if not self.forward_only:
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)

    def schedule_b(self, scheduled_node):
        if not self.forward_only:
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = [None] * len(self.send_tensor_shapes)
            else:
                output_tensor_grad = self.recv_backward_buffer.pop(0)
                for h in output_tensor_grad[1]:
                    for hh in h:
                        hh.wait()
                output_tensor_grad = output_tensor_grad[0]
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'B{scheduled_node.minibatch}')
            if self.run_timer:
                ScheduleTimers.b_cnt += 1
                ScheduleTimers.b.start()
                mem_before = torch.cuda.memory_allocated()
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, self.model_type,
                self.config
            )
            if self.run_timer:
                ScheduleTimers.b.stop()
                ScheduleTimers.b_mem += torch.cuda.memory_allocated() - mem_before
            if get_args().profile:
                torch.cuda.nvtx.range_pop()
            self.send_backward_buffer.append(input_tensor_grad)
            WeightGradStore.flush()

    def schedule_w(self, scheduled_node, non_w_pending):
        if not self.forward_only and non_w_pending:
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'W{scheduled_node.minibatch}')
            if self.run_timer:
                ScheduleTimers.w_cnt += 1
                ScheduleTimers.w.start()
                mem_before = torch.cuda.memory_allocated()
            WeightGradStore.pop()
            if self.run_timer:
                ScheduleTimers.w.stop()
                ScheduleTimers.w_mem += torch.cuda.memory_allocated() - mem_before
            if get_args().profile:
                torch.cuda.nvtx.range_pop()

    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            self.no_sync_context = self.no_sync_func()
            self.no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None

    def prepare(
        self,
        schedule: List[auto_schedule.ScheduledNode],
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ):
        if isinstance(model, list):
            assert (
                len(model) == 1
            ), "non-interleaved pipeline parallelism does not support model chunking"
            model = model[0]
        if isinstance(data_iterator, list):
            assert (
                len(data_iterator) == 1
            ), "non-pipeline-parallel schedule does not support model chunking"
            data_iterator = data_iterator[0]

        config = get_model_config(model)
        if config.overlap_p2p_comm:
            raise ValueError(
                "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
            )
        # Disable async grad reductions
        no_sync_func = config.no_sync_func
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext
        self.no_sync_func = no_sync_func
        self.no_sync_context = None
        if not forward_only:
            ScheduleTimers.iter_counter += 1

        # Checkpoint the activations of partial Transformer layers in a number of micro-batches
        # within the maximum outstanding micro-batch backpropagations.
        # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
        # checkpoint partial Transformer layers (or skip checkpointing) and
        # the rest of micro-batches within a window of micro-batches checkpoint
        # all Transformer layers. The window of micro-batches is set by the maximum
        # outstanding backpropagations and becomes smaller at later pipeline stages.
        # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
        assert config.num_microbatches_with_partial_activation_checkpoints is None

        model_type = get_model_type(model)

        rank = parallel_state.get_pipeline_model_parallel_rank()
        recv_tensor_shapes = get_tensor_shapes(
            rank=rank - 1,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )
        send_tensor_shapes = get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )
        bootstrap_and_profile_p2p_communication(config, send_tensor_shapes,
                                                recv_tensor_shapes)

        run_timer = (
            get_args().zero_bubble_pipeline_timers_end_iter
            >= ScheduleTimers.iter_counter
            >= get_args().zero_bubble_pipeline_timers_start_iter
        )

        self.config = config
        self.model_type = model_type
        self.recv_tensor_shapes = recv_tensor_shapes
        self.send_tensor_shapes = send_tensor_shapes
        self.run_timer = run_timer
        self.schedules = schedule
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.collect_non_loss_data = collect_non_loss_data
        self.forward_only = forward_only
        self._reset()
        self.it = 0

    def run_until_post_validation(self):
        optimizer = self.optimizer
        updated, grad_norm, rollback, succeed = None, None, None, None
        it = 0
        if optimizer.do_this_step:
            assert optimizer.do_prev_step
            if self.data_iterator is not None:
                self.data_iterator.clear_buffer()
                self.data_iterator.save_to_buffer()
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                # print(f"True {rank}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD"]:
                    next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                    next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                    next_compute = next_compute[0] if len(next_compute) > 0 else None
                    self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == 'F':
                    self.schedule_f(scheduled_node)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation()
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert succeed is not None
        else:
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                # print(f"False {rank}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD", "F"]:
                    if optimizer.do_prev_step and scheduled_node.type == "RECV_FORWARD":
                        next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                        next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                        next_compute = next_compute[0] if len(next_compute) > 0 else None
                        self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation()
                    # print(f"{rank} False post validation done")
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert not succeed
        # print(f"{rank}: {optimizer.do_prev_step}, {optimizer.do_this_step} -> {succeed}")
        if not succeed:
            if optimizer.do_prev_step:
                # send dummy recv_forward to clear send_forward request of last rank
                while it < len(self.schedules):
                    scheduled_node = self.schedules[it]
                    if scheduled_node.type == "RECV_FORWARD" and scheduled_node.rollback:
                        # print(f"rollback {rank}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
                        self.add_communication(scheduled_node, False, None)
                    it += 1
            self._reset()
            it = 0
        if succeed and self.data_iterator is not None:
            self.data_iterator.clear_buffer()
        if self.data_iterator is not None:
            self.data_iterator.pop_from_buffer()
        self.it = it
        return updated, grad_norm, rollback

    def run(self):
        self.disable_grad_sync()

        if get_args().profile:
            torch.cuda.nvtx.range_push(f'iter_{torch.distributed.get_rank()}_{ScheduleTimers.iter_counter}')

        it = self.it
        while it < len(self.schedules):
            scheduled_node = self.schedules[it]
            # print(f"iter {torch.distributed.get_rank()}-{it}: {scheduled_node.type}-{scheduled_node.minibatch}")
            if "POST_VALIDATION" in scheduled_node.type:
                pass
            elif scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                next_compute = next_compute[0] if len(next_compute) > 0 else None
                self.add_communication(scheduled_node, next_is_comm, next_compute)
            elif scheduled_node.type == 'F':
                self.schedule_f(scheduled_node)
            elif scheduled_node.type == 'B':
                self.schedule_b(scheduled_node)
            elif scheduled_node.type == 'W':
                non_w_pending = any([node.type != 'W' for node in self.schedules[it + 1:]])
                self.schedule_w(scheduled_node, non_w_pending)
            else:
                raise ValueError(f"Unknown node type {scheduled_node.type}")
            it += 1
        self.it = it

        if get_args().profile:
            torch.cuda.nvtx.range_push('W')
        if not self.forward_only:
            WeightGradStore.clear(self.model)
        if get_args().profile:
            torch.cuda.nvtx.range_pop()  # W
            torch.cuda.nvtx.range_pop()  # iter

        for h in self.send_handles:
            for hh in h:
                for hhh in hh:
                    hhh.wait()

        if not self.forward_only:
            # Launch any remaining grad reductions
            if self.no_sync_context is not None:
                self.enable_grad_sync()

            if self.config.finalize_model_grads_func is not None:
                # Finalize model grads (perform full grad all-reduce / reduce-scatter for
                # data parallelism, layernorm all-reduce for sequence parallelism).
                self.config.finalize_model_grads_func([self.model])

            if get_args().zero_bubble_pipeline_timers_end_iter == ScheduleTimers.iter_counter:
                ScheduleTimers.concluded = True
        return self.forward_data_store

    def __call__(self, *args, **kwargs):
        if kwargs['forward_only']:
            self.prepare(*args, **kwargs)
            assert self.do_post_validation
            self.do_post_validation = True
            self.is_first_run = True
            return self.run()
        if not get_args().enable_optimizer_post_validation:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
            return self.run()
        # enable_optimizer_post_validation == True
        if self.is_first_run:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
        if self.do_post_validation:
            self.prepare(*args, **kwargs)
            result = self.run_until_post_validation()
            self.do_post_validation = False
        else:
            result = self.run()
            self.do_post_validation = True
        return result

zb_v_scheduler = ZeroBubbleVPipeScheduler()
zb_scheduler = ZeroBubbleScheduler()

def get_zb_scheduler_instance():
    if get_args().zero_bubble_v_schedule:
        global zb_v_scheduler
        return zb_v_scheduler
    else:
        global zb_scheduler
        return zb_scheduler


schedule = None
is_auto_schedule = False


def update_schedule(scheduler, f, b, w, c, f_mem, b_mem, w_mem):
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    ag_arguments = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(ag_arguments, (f,b,w,f_mem,b_mem,w_mem))
    assert len(ag_arguments) == torch.distributed.get_world_size()
    f_mid = sorted([x[0] for x in ag_arguments])[len(ag_arguments) // 2]
    b_mid = sorted([x[1] for x in ag_arguments])[len(ag_arguments) // 2]
    w_mid = sorted([x[2] for x in ag_arguments])[len(ag_arguments) // 2]
    f_mem = [x[3] for x in ag_arguments]
    b_mem = [x[4] for x in ag_arguments]
    w_mem = [x[5] for x in ag_arguments]
    if parallel_state.get_pipeline_model_parallel_rank() == 0 and parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
        print(f"rank {torch.distributed.get_rank()} Performing ILP with {f_mid} {b_mid} {w_mid} {c}")
        schedule = scheduler(
            pipeline_model_parallel_size,
            get_num_microbatches(),
            max(int(f_mid * 1000), 1),
            max(int(b_mid * 1000), 1),
            max(int(w_mid * 1000), 1),
            max(int(c * 1000), 1),
            f_mem, b_mem, w_mem,
            get_args().zero_bubble_max_pending_backward
        )
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, schedule)
    else:
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, None)
        schedule = list(filter(lambda x: x is not None, ag_result))
        assert len(schedule) == 1
        schedule = schedule[0]
    return schedule


def get_zero_bubble_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pipeline_model_parallel_size > 1, "zero bubble must be enabled with pipeline parallelism"

    args = get_args()
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    seq_length = args.seq_length
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx

    final_layer_additional_mem = 2 * get_tokenizer().vocab_size

    # f_mem_array = [f_mem] * pipeline_model_parallel_size
    # b_mem_array = [b_mem] * pipeline_model_parallel_size
    # w_mem_array = [w_mem] * pipeline_model_parallel_size
    # if not args.zero_bubble_v_schedule:
    #     f_mem_array[0] += final_layer_additional_mem
    #     w_mem_array[0] -= final_layer_additional_mem

    def wrapped_auto_schedule_forward_backward_func(func, scheduler):
        global schedule, is_auto_schedule
        if schedule is None:
            schedule = update_schedule(scheduler,
                f=1,
                b=1,
                w=1,
                c=0,
                f_mem=f_mem_approx,
                b_mem=b_mem_approx,
                w_mem=w_mem_approx)
        if ScheduleTimers.concluded and not is_auto_schedule:
            conclusion = ScheduleTimers.conclusion()
            # TODO(wanxy): Maybe an all-reduce here to collect global stats?
            print(f"rank {torch.distributed.get_rank()} profiling conclusion: {conclusion}")
            schedule = update_schedule(scheduler,
                *conclusion)
            is_auto_schedule = True

        def wrap_schedule(**kwargs):
            return func(
                schedule=schedule[parallel_state.get_pipeline_model_parallel_rank()], **kwargs
            )
        return wrap_schedule

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem):
            return v_schedule.PipelineGraph(
                nstages,
                nmb,
                f,b,w,c,
                f_mem=f_mem[0], b_mem=b_mem[0], w_mem=w_mem[0],
                max_mem=None
                # Mem ignored for now
            ).get_v_schedule()
        if get_args().zero_bubble_v_schedule:
            global_zb_scheduler = get_zb_scheduler_instance()
            forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_scheduler, scheduler=scheduler)
            # forward_backward_func = wrapped_auto_schedule_forward_backward_func(forward_backward_pipelining_with_interleaving_auto_schedule,
            #                                                                     scheduler=scheduler)
        else:
            raise ValueError("got virtual pipeline parallel but v_schedule is disabled")
    else:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem):
            return auto_schedule.auto_schedule(
                nstages,
                nmb,
                auto_schedule.GraphConfig(
                    cost_f=f,
                    cost_b=b,
                    cost_w=w,
                    cost_comm=c,
                    mem_f=f_mem[0],
                    mem_b=b_mem[0],
                    mem_w=w_mem[0],
                    max_mem=mem * f_mem[0],
                    print_scaling=1000
                ),
            )

        global_zb_scheduler = get_zb_scheduler_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_scheduler, scheduler=scheduler)

    return forward_backward_func
