import argparse

def get_params():
    parser = argparse.ArgumentParser(description="NER Task")

    parser.add_argument("--exp_name", type=str, default="conll2003", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="train.log")

    parser.add_argument("--dump_path", type=str, default="logs", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")

    parser.add_argument("--model_name", type=str, default="roberta-large", help="model name")
    parser.add_argument("--seed", type=int, default=111, help="random seed")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--early_stop", type=int, default=3, help="No improvement after several epoch, we stop training")
    parser.add_argument("--num_tag", type=int, default=3, help="Number of entity in the dataset")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden layer dimension")
    parser.add_argument("--data_folder", type=str, default="/gpfs/fs1/projects/gpu_adlr/datasets/zihanl/conll2003", help="NER data folder")
    parser.add_argument("--saved_folder", type=str, default="/gpfs/fs1/projects/gpu_adlr/datasets/zihanl/checkpoints/ner_model", help="NER data folder")

    params = parser.parse_args()

    return params
