import os
import argparse


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='mixformer_vit', help='training script name')
    parser.add_argument('--config', type=str, default='baseline_large', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, default='mixformer_train', help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="single",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', default=0, type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--master_port', type=int, help="master port", default=26500)
   
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s " \
                    % (args.script, args.config, args.save_dir)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    print(train_cmd)
    os.system(train_cmd)


if __name__ == "__main__":
    main()
