# train.py, will train a new network on a dataset and save the model as a checkpoint.
# Train a new network on a data set with train.py
# Basic usage: `python train.py data_directory`
# Options:
#     --save_dir: Set directory to save checkpoints `python train.py data_dir --save_dir save_directory`
#     --arch: Choose base architecture `python train.py data_dir --arch "vgg13"
#     --learning_rate, --hidden_units, --epochs: Used to control hyperparameters
#         `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
#     --gpu: Train on GPU instead of CPU `python train.py data_dir --gpu`

import argparse
import os
import torch
from model import Model


def main(data_dir: str, save_dir: str, arch: str, learning_rate: float, hidden_units: int, epochs: int, gpu: bool):
    assert os.path.isdir(data_dir), f"Directory {data_dir} does not exist"
    assert not gpu or torch.cuda.is_available(), "CUDA not available"
    model = Model(gpu, save_dir=save_dir, epochs=epochs)
    model.new(arch, learning_rate, hidden_units, 102)
    model.load_data(data_dir)
    model.train()
    model.test()


def parser():
    parser = argparse.ArgumentParser(description='Train a neural network on image data')
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default=None, help='Set directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Choose base architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Set the learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Set the number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='Set the number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Train on GPU instead of CPU')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    print(args)
    main(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
