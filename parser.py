import argparse

import torch

parser = argparse.ArgumentParser()

parser.add_argument('--type', default="train")

parser.add_argument('--train_path', default="data/train/")
parser.add_argument('--test_path', default="data/test/")

parser.add_argument('--save_model', default="model_checkpoint/best_model.pth")
parser.add_argument('--save_loss_path', default="data/loss.txt")

args = parser.parse_args()
