import torch
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.distributed
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import print_rank_0, get_tokenizer, get_args
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.utils import divide
from megatron.model import GPTModelPipe, Float16Module
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.arguments import core_transformer_config_from_args
from megatron.initialize import initialize_megatron
from megatron.optimizer import get_megatron_optimizer
from megatron.checkpointing import save_checkpoint, load_checkpoint
from megatron.training import get_optimizer_param_scheduler
from deepspeed.runtime.utils import see_memory_usage
import deepspeed
import copy
from pathlib import Path



def add_extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='hf2mega')
    group.add_argument("--hf-ckpt-dir",
                       type=str,
                       default="",
                       help="the llama-hf ckpt")
    group.add_argument("--hf-ckpt-num-shards", type=int, default=-1, help='num of llama ckpt.')
    group.add_argument("--load-mode", type=str,
                       default=None,
                       choices=['torchbin', 'safetensor', 'auto'],
                       help="load ckpt format: pytorch.bin or model.safetensor or auto.")
    group.add_argument("--to-hf-ckpt", action="store_true",
                       help="by default convert from hf to megads"
                            "if set, convert reversely from megads to hf ckpt.")
    return parser


def compute_partition_range(hidden_size, local_rank, tp_size):
    partition_size = divide(hidden_size, tp_size)
    start_index = local_rank * partition_size
    end_index = start_index + partition_size
    return partition_size, start_index, end_index


def load_and_print_hf_weight(hf_ckpt_dir, hf_ckpt_num_of_shards):
    # Optimization point: We can selectively load specific 'shared' data to reduce CPU memory usage.
    loaded = {}
    print_rank_0(
        f"----------------------------hf weight list----------------------------")

    for wid in range(1, hf_ckpt_num_of_shards + 1):
        d = torch.load(
            f"{hf_ckpt_dir}/pytorch_model-{wid:05d}-of-{hf_ckpt_num_of_shards:05d}.bin",
            map_location=torch.device('cpu'))
        for k in d:
            print_rank_0(k)
            assert k not in loaded
            loaded[k] = d[k].clone()
    del d
    return loaded


def load_and_print_hf_weight_from_safetensor(hf_ckpt_dir, hf_ckpt_num_of_shards):
    from safetensors import safe_open
    # Optimization point: We can selectively load specific 'shared' data to reduce CPU memory usage.
    hf_model = {}
    print_rank_0(
        f"----------------------------hf weight list----------------------------")

    for wid in range(1, hf_ckpt_num_of_shards + 1):
        if hf_ckpt_num_of_shards == 1:
            ckpt_path = f"{hf_ckpt_dir}/model.safetensors"
        else:
            ckpt_path = f"{hf_ckpt_dir}/model-{wid:05d}-of-{hf_ckpt_num_of_shards:05d}.safetensors"

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                print_rank_0(f"name: {k}, shape: {f.get_tensor(k).shape}")
                assert k not in hf_model
                hf_model[k] = f.get_tensor(k).clone()

    return hf_model


def load_and_print_hf_weight_auto(hf_ckpt_dir, no_init=True):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights

    if no_init:
        hf_config = AutoConfig.from_pretrained(hf_ckpt_dir, trust_remote_code=True)
        with no_init_weights():
            hf_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    else:
        hf_model = {}
        hf_auto_model = AutoModelForCausalLM.from_pretrained(hf_ckpt_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
        print_rank_0(
            f"----------------------------hf weight list----------------------------")

        for name, param in hf_auto_model.named_parameters():
            hf_model[name] = param.clone()
            print_rank_0(name)

    return hf_model


def print_distinct_weights(model):
    print_rank_0(
        f"----------------------------mega-ds weight list----------------------------")
    for pipe_rank in range(mpu.get_pipeline_model_parallel_world_size()):
        if mpu.get_pipeline_model_parallel_rank() == pipe_rank:
            if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank(
            ) == 0:
                for pname, p in model.named_parameters():
                    print(pname)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()


class refactor:
    def __init__(self, ds_model, hf_model, args, config):
        tokenizer = get_tokenizer()
        # align layer number
        self.ds_model = ds_model
        self.hf_model = hf_model
        self.hf_dict = {} # for handling pp case when converting mds => hf
        self.config = config

        self.offset_num = 2
        self.mega_emb_wnum = 1
        self.mega_norm_wnum = args.num_layers + 2
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.mega_lm_head_wnum = self.mega_norm_wnum + 1
        self.token_vocab = tokenizer.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.more_padded = self.padded_vocab_size - self.token_vocab
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.decoder_pat = re.compile("(\d+)\.(.+)")
        self.refactor_weight_list = []
        self.is_refactored = False

    def _embedding_refactor(self, pname, p):
        if pname == f"{self.mega_lm_head_wnum}.lm_head.weight":
            hf_name = "lm_head.weight"
        elif pname == f"{self.mega_emb_wnum}.word_embeddings.weight":
            hf_name = "model.embed_tokens.weight"
        hf_w = self.hf_model[hf_name]
        assert hf_w.shape[0] == self.token_vocab
        per_partition_vocab_size, start_index, end_index = compute_partition_range(
            self.padded_vocab_size, self.tp_rank, self.tp_size)
        end_index = min(end_index, self.token_vocab)
        real_partition_vocab_size = end_index - start_index

        new_w = torch.zeros((per_partition_vocab_size, hf_w.shape[1]), dtype=hf_w.dtype)
        new_w[:real_partition_vocab_size, :] = hf_w[start_index:end_index, :]
        if self.tp_rank == self.tp_size - 1 and self.more_padded > 0:
            new_w[-self.more_padded:] = hf_w[:self.token_vocab].mean(dim=0, keepdim=True)

        self.record_mapping_info(
            f"mega-ds: {pname,p.data.shape}<--hf: {hf_name,}  [{start_index}:{end_index},:]  of {hf_w.shape}"
        )
        return new_w

    


    def _direct_refactor(self, pname, p, hf_layer=None, subname=None):
        if pname == f"{self.mega_norm_wnum}.weight":
            hf_name = "model.norm.weight"
        elif subname in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            hf_name = f"model.layers.{hf_layer}.{subname}"

        new_w = hf_w = self.hf_model[hf_name]
        self.record_mapping_info(
            f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  {hf_w.shape}")
        return new_w


    def _qkv_refactor(self, pname, p, hf_layer):
        hf_wq_name = f"model.layers.{hf_layer}.self_attn.q_proj.weight"
        hf_wk_name = f"model.layers.{hf_layer}.self_attn.k_proj.weight"
        hf_wv_name = f"model.layers.{hf_layer}.self_attn.v_proj.weight"
        wq = self.hf_model[hf_wq_name]
        wk = self.hf_model[hf_wk_name]
        wv = self.hf_model[hf_wv_name]

        query_hidden_size = wq.shape[0]
        kv_hidden_size = wk.shape[0]

        per_partition_size, start_qindex, end_index = compute_partition_range(
            query_hidden_size, self.tp_rank, self.tp_size)
        _,start_kvindex, _= compute_partition_range(
            kv_hidden_size, self.tp_rank, self.tp_size)

        hidden_size_per_attention_head = divide(query_hidden_size,
                                                self.config.num_attention_heads)
        num_attention_heads_per_partition = divide(self.config.num_attention_heads,
                                                   self.tp_size)

        num_kv_heads_per_partition= divide(self.config.num_key_value_heads,
                                                   self.tp_size)
        qkv_size=(num_attention_heads_per_partition+2*num_kv_heads_per_partition)*hidden_size_per_attention_head
        num_qheads_per_group=divide(self.config.num_attention_heads,self.config.num_key_value_heads)
        num_groups =divide(num_attention_heads_per_partition,num_qheads_per_group)
        new_w = torch.zeros((qkv_size, wq.shape[1]), dtype=wq.dtype)

        for i in range(num_groups):
            query_current_index=start_qindex+i*num_qheads_per_group*hidden_size_per_attention_head
            query_next_index=query_current_index+num_qheads_per_group*hidden_size_per_attention_head
            kv_current_index=start_kvindex+i*hidden_size_per_attention_head
            kv_next_kvindex=kv_current_index+hidden_size_per_attention_head

            new_w_index=i* (num_qheads_per_group+2)*hidden_size_per_attention_head

            new_w[new_w_index:new_w_index+(num_qheads_per_group+2)*hidden_size_per_attention_head,:]=\
                torch.cat([
                    wq[query_current_index:query_next_index,:],
                    wk[kv_current_index:kv_next_kvindex,:],
                    wv[kv_current_index:kv_next_kvindex,:]
                ],dim=0)

        self.record_mapping_info(
            f"mega-ds:{pname,p.data.shape}<--hf{hf_wq_name,hf_wk_name,hf_wv_name,}  cat q,k,v [{query_current_index}:{query_next_index},:]  of q,k,v{wq.shape}"
        )
        return new_w

    def _mlphto4h_dense_refactor(self, pname, p, hf_layer):
        hf_w_gate_name = f"model.layers.{hf_layer}.mlp.gate_proj.weight"
        hf_w_up_name = f"model.layers.{hf_layer}.mlp.up_proj.weight"
        w_gate = self.hf_model[hf_w_gate_name]
        w_up = self.hf_model[hf_w_up_name]

        hidden_size = w_gate.shape[0]
        per_partition_size, start_index, end_index = compute_partition_range(
            hidden_size, self.tp_rank, self.tp_size)
        new_w = torch.zeros((per_partition_size * 2,
                             w_gate.shape[1]),
                            dtype=w_gate.dtype)
        new_w[:per_partition_size * 2, :] = \
                torch.cat([
                    w_gate[start_index:end_index, :],
                    w_up[start_index:end_index, :]
                ], dim=0)
        self.record_mapping_info(
            f"mega-ds:{pname,p.data.shape}<--hf{hf_w_gate_name,hf_w_up_name}  cat gate,up [{start_index}:{end_index},:]  of gate,up{w_gate.shape}"
        )
        return new_w

    def _attn_dense_refactor(self, pname, p, hf_layer, subname):
        if subname == "self_attention.dense.weight":
            hf_name = f"model.layers.{hf_layer}.self_attn.o_proj.weight"
        else:
            hf_name = f"model.layers.{hf_layer}.mlp.down_proj.weight"

        hf_w = self.hf_model[hf_name]
        hidden_size = hf_w.shape[1]
        per_partition_size, start_index, end_index = compute_partition_range(
            hidden_size, self.tp_rank, self.tp_size)
        new_w = torch.zeros((hf_w.shape[0], per_partition_size), dtype=hf_w.dtype)
        new_w[:, :per_partition_size] = hf_w[:, start_index:end_index]
        self.record_mapping_info(
            f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  [:,{start_index}:{end_index}]  of {hf_w.shape}"
        )
        return new_w

    def _mlphto4h1_refactor(self, pname, p, hf_layer, subname):
        if subname == "mlp.dense_h_to_4h1.weight":
            hf_name = f"model.layers.{hf_layer}.mlp.gate_proj.weight"
        else:
            hf_name = f"model.layers.{hf_layer}.mlp.up_proj.weight"
        hf_w = self.hf_model[hf_name]
        hidden_size = hf_w.shape[0]
        per_partition_size, start_index, end_index = compute_partition_range(
            hidden_size, self.tp_rank, self.tp_size)
        new_w = torch.zeros((per_partition_size, hf_w.shape[1]), dtype=hf_w.dtype)

        new_w[:per_partition_size, :] = hf_w[start_index:end_index, :]
        self.record_mapping_info(
            f"mega-ds:{pname,p.data.shape}<--hf{hf_name,}  [{start_index}:{end_index},:]  of {hf_w.shape}"
        )
        return new_w

    def transform_from_hf_to_megds(self):
        assert self.is_refactored == False
        new_w = None
        for pname, p in self.ds_model.named_parameters():

            if pname in [
                    f"{self.mega_emb_wnum}.word_embeddings.weight",
                    f"{self.mega_lm_head_wnum}.lm_head.weight"
            ]:
                new_w = self._embedding_refactor(pname, p)
            elif pname == f"{self.mega_norm_wnum}.weight":
                new_w = self._direct_refactor(pname, p)
            else:
                mobj = self.decoder_pat.match(pname)
                layer_num = int(mobj.group(1))
                subname = mobj.group(2)
                hf_layer = layer_num - self.offset_num
                if subname in ["self_attention.query_key_value.weight"]:
                    new_w = self._qkv_refactor(pname, p, hf_layer)
                elif subname in ["mlp.dense_h_to_4h.weight"]:
                    new_w = self._mlphto4h_dense_refactor(pname, p, hf_layer)
                elif subname in [
                        "self_attention.dense.weight",
                        "mlp.dense_4h_to_h.weight"
                ]:
                    new_w = self._attn_dense_refactor(pname, p, hf_layer, subname)
                elif subname in [
                        "mlp.dense_h_to_4h1.weight",
                        "mlp.dense_h_to_4h2.weight"
                ]:
                    new_w = self._mlphto4h1_refactor()
                elif subname in [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight"
                ]:
                    new_w = self._direct_refactor(pname, p, hf_layer, subname)
                else:
                    raise ValueError("Unrecognized weight type")
            p.data.copy_(new_w)
            new_w = None
        self.is_refactored = True

    
    def _embedding_refactor_to_hf(self, pname, ds_w):
        if pname == f"{self.mega_lm_head_wnum}.lm_head.weight":
            hf_w = self.hf_model.lm_head.weight
            hf_w_name = "lm_head.weight"
        elif pname == f"{self.mega_emb_wnum}.word_embeddings.weight":
            hf_w = self.hf_model.model.embed_tokens.weight
            hf_w_name = "model.embed_tokens.weight"

        with torch.no_grad():
            ds_w_all_rank = tensor_parallel.mappings._gather_along_first_dim(ds_w)
        
        self.hf_dict[hf_w_name] = copy.deepcopy(ds_w_all_rank[:hf_w.shape[0], :])

    def _direct_refactor_to_hf(self, pname, ds_w, hf_layer=None, subname=None):
        if pname in [f"{self.mega_norm_wnum}.weight"]:
            hf_w = self.hf_model.model.norm.weight
            hf_w_name = "model.norm.weight"
        elif subname in ["input_layernorm.weight"]:
            hf_w = self.hf_model.model.layers[hf_layer].input_layernorm.weight
            hf_w_name = f"model.layers.{hf_layer}.input_layernorm.weight"
        elif subname in ["post_attention_layernorm.weight"]:
            hf_w = self.hf_model.model.layers[hf_layer].post_attention_layernorm.weight
            hf_w_name = f"model.layers.{hf_layer}.post_attention_layernorm.weight"

        self.hf_dict[hf_w_name] = copy.deepcopy(ds_w)

    def _attn_dense_refactor_to_hf(self, pname, ds_w, hf_layer, subname):
        if subname == "self_attention.dense.weight":
            hf_w = self.hf_model.model.layers[hf_layer].self_attn.o_proj.weight
            hf_w_name = f"model.layers.{hf_layer}.self_attn.o_proj.weight"
        elif subname == "mlp.dense_4h_to_h.weight":
            hf_w = self.hf_model.model.layers[hf_layer].mlp.down_proj.weight
            hf_w_name = f"model.layers.{hf_layer}.mlp.down_proj.weight"

        with torch.no_grad():
            ds_w_all_rank = tensor_parallel.mappings._gather_along_last_dim(ds_w)

        self.hf_dict[hf_w_name] = copy.deepcopy(ds_w_all_rank)

    def _mlphto4h_dense_refactor_to_hf(self, pname, ds_w, hf_layer):
        hf_g_name = f"model.layers.{hf_layer}.mlp.gate_proj.weight"
        hf_u_name = f"model.layers.{hf_layer}.mlp.up_proj.weight"
        
        with torch.no_grad():
            ds_w_all_rank = tensor_parallel.mappings._gather_along_first_dim(ds_w)
        
        ds_w_shape = ds_w_all_rank.shape
        ds_w_all_rank = ds_w_all_rank.reshape(self.tp_size, 2, -1, ds_w_shape[-1])
        self.hf_dict[hf_g_name] = copy.deepcopy(ds_w_all_rank[:, 0, :, :].reshape(-1, ds_w_shape[-1]))
        self.hf_dict[hf_u_name] = copy.deepcopy(ds_w_all_rank[:, 1, :, :].reshape(-1, ds_w_shape[-1]))

    
    def _qkv_refactor_to_hf(self, pname, ds_w, hf_layer):
        with torch.no_grad():
            ds_w_all_rank = tensor_parallel.mappings._gather_along_first_dim(ds_w)

        hf_q = self.hf_model.model.layers[hf_layer].self_attn.q_proj.weight
        hf_k = self.hf_model.model.layers[hf_layer].self_attn.k_proj.weight
        hf_v = self.hf_model.model.layers[hf_layer].self_attn.v_proj.weight
        hf_q_name = f"model.layers.{hf_layer}.self_attn.q_proj.weight"
        hf_k_name = f"model.layers.{hf_layer}.self_attn.k_proj.weight"
        hf_v_name = f"model.layers.{hf_layer}.self_attn.v_proj.weight"
        oldshape = hf_q.shape
        hidden_size = oldshape[-1]
        hidden_size_per_attention_head = divide(hidden_size,
                                                self.config.num_attention_heads)
        # MHA & GQA
        group = divide(self.config.num_attention_heads, self.config.num_key_value_heads)
        newshape = (self.config.num_key_value_heads, group + 2, hidden_size_per_attention_head, hidden_size)
        ds_w_out = ds_w_all_rank.reshape(*newshape)
        query_weight, key_weight, value_weight = torch.split(ds_w_out, [group, 1, 1], dim=1)
        self.hf_dict[hf_q_name] = copy.deepcopy(query_weight.reshape(-1, hidden_size))
        self.hf_dict[hf_k_name] = copy.deepcopy(key_weight.reshape(-1, hidden_size))
        self.hf_dict[hf_v_name] = copy.deepcopy(value_weight.reshape(-1, hidden_size))
        del query_weight, key_weight, value_weight


    def transform_from_megads_to_hf(self):

        for pname, p in self.ds_model.named_parameters():
            if pname in [
                    f"{self.mega_emb_wnum}.word_embeddings.weight",
                    f"{self.mega_lm_head_wnum}.lm_head.weight",
            ]:
                self._embedding_refactor_to_hf(pname, p)
            elif pname in [
                    f"{self.mega_norm_wnum}.weight",
            ]:
                self._direct_refactor_to_hf(pname, p)
            else:
                mobj = self.decoder_pat.match(pname)
                layer_num = int(mobj.group(1))
                subname = mobj.group(2)
                hf_layer = layer_num - self.offset_num
                if subname in ["self_attention.query_key_value.weight"]:
                    self._qkv_refactor_to_hf(pname, p, hf_layer)
                elif subname in ["mlp.dense_h_to_4h.weight"]:
                    self._mlphto4h_dense_refactor_to_hf(pname, p, hf_layer)
                elif subname in [
                    "self_attention.dense.weight",
                    "mlp.dense_4h_to_h.weight"
                ]:
                    self._attn_dense_refactor_to_hf(pname, p, hf_layer, subname)
                elif subname in [
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                ]:
                    self._direct_refactor_to_hf(pname, p, hf_layer, subname)
                else:
                    print(f"Unrecognized weight type: {pname}")
                    raise ValueError(f"Unrecognized weight type: {pname}")
        self.is_refactored = True

    def record_mapping_info(self, record_msg):
        self.refactor_weight_list.append(record_msg)

    def inorder_show_record(self):
        assert self.is_refactored
        print_rank_0(
            f"----------------------------mapping list----------------------------")
        # print dp rank0 tp rank0  records.
        for pipe_rank in range(mpu.get_pipeline_model_parallel_world_size()):
            if mpu.get_pipeline_model_parallel_rank() == pipe_rank:
                if mpu.get_data_parallel_rank(
                ) == 0 and mpu.get_tensor_model_parallel_rank() == 0:
                    for record in self.refactor_weight_list:
                        print(record)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()


def load_hf_weights(args, no_init):
    if args.load_mode == 'torchbin':
        assert no_init == False, "only work with init"
        return load_and_print_hf_weight(args.hf_ckpt_dir, args.hf_ckpt_num_shards)
    elif args.load_mode == 'safetensor':
        assert no_init == False, "only work with init"
        return load_and_print_hf_weight_from_safetensor(args.hf_ckpt_dir, args.hf_ckpt_num_shards)
    elif args.load_mode ==  'auto':
        return load_and_print_hf_weight_auto(args.hf_ckpt_dir, no_init)


def convert_ckpt():
    """Build the model."""
    args = get_args()
    print_rank_0(f'building model ...')
    see_memory_usage(f"Before Building Model", force=True)

    config = core_transformer_config_from_args(args)
    with deepspeed.zero.Init(
            data_parallel_group=mpu.get_data_parallel_group(),
            remote_device=None if args.remote_device == 'none' else args.remote_device,
            config_dict_or_path=args.deepspeed_config,
            enabled=args.zero_stage == 3,
            mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            ds_model = GPTModelPipe(config, num_tokentypes=0, parallel_output=True)
        else:
            raise NotImplementedError("Not implemented")

    see_memory_usage(f"After Building Model", force=True)
    if torch.distributed.get_rank() < 2:
        print(f"{torch.distributed.get_rank()} {ds_model}")

    # 'torchbin', 'safetensor', 'auto'
    hf_model = load_hf_weights(args, no_init=args.to_hf_ckpt)

    # print_distinct_weights(hf_model)

    #init model and save
    print_rank_0(f"before deepspeed init")
    ds_engine, _, _, _ = deepspeed.initialize(
        model=ds_model,
        optimizer=None,
        args=args,
        lr_scheduler=None,
        mpu=mpu if args.no_pipeline_parallel else None)
    print_rank_0(f"after deepspeed init")

    if args.to_hf_ckpt:
        load_checkpoint([ds_engine], None, None, load_only_weights=True)
        print_rank_0(f"completed to load deepspeed actual checkpoint")

    # refactor weight from hf to mega-ds and vice versa

    cur_refactor = refactor(ds_model, hf_model, args, config)
    if args.to_hf_ckpt:
        cur_refactor.transform_from_megads_to_hf()
    else:
        cur_refactor.transform_from_hf_to_megds()
    # cur_refactor.inorder_show_record()

    if args.to_hf_ckpt:
        save_path = args.save
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)
        ckpt_per_pp_path = os.path.join(save_path, f"model_pp{mpu.get_pipeline_model_parallel_rank()}.pt")
        torch.save(cur_refactor.hf_dict, ckpt_per_pp_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print_rank_0(f"hf checkpoint will be saved in {save_path}/release ")
        if mpu.is_pipeline_last_stage():
            ## doing checkpoint merging and saving...
            # hf_model.tie_weights()

            all_wei = {}
            for pprank in range(mpu.get_pipeline_model_parallel_world_size()):
                ckpt_per_pp_path = os.path.join(save_path, f"model_pp{pprank}.pt")
                partial_wei = torch.load(ckpt_per_pp_path)
                all_wei = all_wei | partial_wei

            hf_model.load_state_dict(all_wei)

            # mega-ds checkpoint will be saved in  args.save
            hf_model.save_pretrained(os.path.join(save_path, "release"), safe_serialization=True)
    else:
        print_rank_0(f"mega-ds checkpoint will be saved in {args.save}")
        save_checkpoint(0, [ds_engine], None, None)
    
    print_rank_0(f"save checkpoint completed")

if __name__ == "__main__":

    initialize_megatron(extra_args_provider=add_extra_args)
    convert_ckpt()
