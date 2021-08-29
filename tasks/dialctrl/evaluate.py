
from megatron import get_args
from megatron import get_timers
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.training import evaluate_and_print_results
from megatron.training import setup_model_and_optimizer
from megatron.checkpointing import load_checkpoint
from tasks.finetune_utils import build_data_loader
from tasks.dialctrl.data import build_test_dataset
from tasks.dialctrl.finetune import model_provider, process_batch, loss_func, forward_step
from tasks.dialctrl.metrics import F1Metric
from tqdm import tqdm

def test_dataset_provider():
    """Build the test dataset for dialog/control module"""
    args = get_args()
    print_rank_0('> building the test dataset for %s module ...' \
                    % args.train_module)

    test_ds = build_test_dataset(
        test_data_path=args.test_data_path,
        train_module=args.train_module,
        max_seq_len=args.max_seq_len,
        last_turn=args.last_turn,
        no_control_code=args.no_control_code,
        add_separator=args.add_separator,
        add_ctrl_code_to_dialog=args.add_ctrl_code_to_dialog,
        remove_ctrl_sent=args.remove_ctrl_sent)

    print_rank_0("> finished creating the test dataset for %s module ..." \
                    % args.train_module)

    print_rank_0('> test set size: %d' % len(test_ds))
    args.eval_iters = len(test_ds) // args.global_batch_size
    print_rank_0('> evaluation iteration: %d' % args.eval_iters)

    return test_ds


def _build_test_iterator(test_dataset, task_collate_fn=None):
    """Test dataloader."""
    args = get_args()

    print_rank_0('building test dataloader ...')
    # Test loader
    test_dataloader = build_data_loader(test_dataset, args.micro_batch_size,
                                        args.num_workers, not args.keep_last,
                                        task_collate_fn)
    test_iterator = test_dataloader.__iter__()
    return test_iterator


def evaluate_ppl(test_dataset_provider, model_provider, forward_step):
    args = get_args()
    timers = get_timers()

    # test dataloader.
    timers('test dataset/dataloder').start()
    test_dataset = test_dataset_provider()
    test_iterator = _build_test_iterator(test_dataset)
    timers('test dataset/dataloder').stop()

    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    timers('pretrained checkpoint').start()
    if args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        original_rng = args.no_load_rng
        args.no_load_rng = True
        iteration = load_checkpoint(model, None, None)
        args.load = original_load
        args.no_load_rng = original_rng
        # This is critical when only model is loaded. We should make sure
        # main parameters are also updated.
        optimizer.reload_model_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['test dataset/dataloder', 'model and optimizer', 
                'pretrained checkpoint'])
    
    print_rank_0('evaluating ...')
    prefix = 'iteration {}'.format(iteration)
    evaluate_and_print_results(prefix, forward_step, 
                               test_iterator, model,
                               iteration, False)
    
    print_rank_0('done :-)')


def evaluate_f1(guess_file, answer_file, remove_stopwords):

    guess_list = []
    print_rank_0('reading %s' % guess_file)
    with open(guess_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            guess_list.append(line)

    answer_list = []
    print_rank_0('reading %s' % answer_file)
    with open(answer_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if line == "no_passages_used":
                line = ""
            answer_list.append(line)

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list, remove_stopwords)
    print_rank_0('Precision: %.4f; recall: %.4f; f1: %.4f' % (precision, recall, f1))

    print_rank_0('done :-)')


def main():
    args = get_args()

    if 'ppl' in args.task: 
        evaluate_ppl(test_dataset_provider, model_provider, forward_step)
    
    elif 'f1' in args.task:
        evaluate_f1(args.guess_file, args.answer_file, args.remove_stopwords)

