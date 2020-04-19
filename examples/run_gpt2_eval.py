"""
example usage:
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --model-path <gpt2_117_path> \
  --data-path <wikitext_tokens_test_path> \
  --batch-size 16 \
  --cache-dir <cache dir path>
"""
import argparse
import subprocess

parser = argparse.ArgumentParser('run zero shot GPT2 eval')
parser.add_argument('--model-path', type=str, required=True,
                    help='Saved model path for evaluation')
parser.add_argument('--batch-size', type=int, default=4,
                    help='batch size to use for evaluation')
parser.add_argument('--num-attention-heads', type=int, default=12,
                    help='num of transformer attention heads')
parser.add_argument('--hidden-size', type=int, default=768,
                    help='tansformer hidden size')
parser.add_argument('--num-layers', type=int, default=12,
                    help='num decoder layers')
parser.add_argument('--data-path', type=str, required=True,
                    help='Data path for evaluation data')
parser.add_argument('--cloze-eval', action='store_true',
                    help='Run lambada cloze eval instead of perplexity eval.')
parser.add_argument('--easy-lambada', action='store_true',
                       help='use easier formulation of lambada')
parser.add_argument('--model-parallel-size', type=int, default=1,
                    help='model parallel size to use')
args = parser.parse_args()

multinode_args = ''
if args.model_parallel_size > 1:
    multinode_args += ' -m torch.distributed.launch --nproc_per_node {} '.format(args.model_parallel_size)

CMD = ' --model-parallel-size {model_par} \
       --num-layers {nlayers} \
       --hidden-size {hidden} \
       --log-interval 100 \
       --load {model} \
       --batch-size {batch} \
       --num-attention-heads {natt} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --distributed-backend nccl \
       --hidden-dropout 0.1 \
       --attention-dropout 0.1 \
       --fp16 \
       --lr 1 --no-load-optim --no-load-rng --epochs 0 \
       --overlapping-eval 32 \
       --merge-file /home/universal-lm-data.cosmos549/repos/megatron_latest/vocab_cache/merges.txt \
       --vocab-file /home/universal-lm-data.cosmos549/repos/megatron_latest/vocab_cache/vocab.json'.format(model_par=args.model_parallel_size,
                                    nlayers=args.num_layers,
                                    hidden=args.hidden_size,
                                    model=args.model_path,
                                    batch=args.batch_size,
                                    natt=args.num_attention_heads,)

if args.cloze_eval:
    CMD += ' --valid-data {} '.format(args.data_path)
    CMD += ' --task LAMBADA '
    if not args.easy_lambada:
      CMD += ' --strict-lambada '
    CMD = 'main.py' + CMD
    print('Running Lambada Eval Command:', flush=True)
else:
    CMD += ' --valid-data {} '.format(args.data_path)
    CMD += ' --task WIKITEXT103 '
    CMD = 'main.py' + CMD
    print('Running PPL Eval Command:', flush=True)

CMD = 'python3 '+multinode_args+CMD
print(CMD, flush=True)

subprocess.call(CMD.split())
