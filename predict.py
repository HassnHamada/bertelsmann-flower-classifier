
# predict.py, uses a trained network to predict the class for an input image.
# Predict flower name from an image with predict.py along with the probability of that name.
# Basic usage: python predict.py /path/to/image checkpoint
# Options:
#   --top_k: Return top K most likely classes `python predict.py input checkpoint --top_k 3`
#   --category_names: Use to map of categories to real names `python predict.py input checkpoint --category_names cat_to_name.json`
#   --gpu: Use GPU for inference `python predict.py input checkpoint --gpu`

import argparse
import os
import torch
from model import Model
import json


def main(checkpoint: str, image: str, top_k: int, cat2name: float, gpu: bool):
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} doesn't exist"
    assert os.path.exists(image), f"Image {image} doesn't exist"
    assert cat2name is None or os.path.exists(cat2name), f"Categories {cat2name} doesn't exist"
    assert not gpu or torch.cuda.is_available(), "CUDA not available"
    model = Model(gpu)
    model.load(checkpoint)
    prob, pred = model.predict(image, top_k)
    if cat2name is not None:
        with open(cat2name, 'r') as f:
            cat2name = json.load(f)
        pred = [cat2name[i] for i in pred]
    print(*(f"Class {repr(i)} with likely  {j:.6f}" for i, j in zip(pred, prob)), sep='\n')
    return prob, pred


def parser():
    parser = argparse.ArgumentParser(description='Predict image label')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model file')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    return args


def test():
    model = Model(True)
    model.load(r'checkpoints\best.pt')
    total = corect = 0
    for cls in os.listdir('./flowers/test'):
        for img in os.listdir(f'./flowers/test/{cls}'):
            prob, pred = model.predict(f'./flowers/test/{cls}/{img}')
            print(f"{cls}: {img} -> {pred[0]} with {prob[0]:.6f}")
            total += 1
            if pred[0] == cls:
                corect += 1
    print(f"Correct: {corect} / {total} = {corect / total}")


if __name__ == '__main__':
    args = parser()
    print(args)
    main(args.checkpoint, args.image, args.top_k, args.category_names, args.gpu)
