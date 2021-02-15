import os
import json
from glob import glob
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import argparse
from dataloader import get_dataloader
from models.protonet import ProtoNet
from models.resnet import ResNet
from prototypical_loss import PrototypicalLoss

from utils.train_utils import AverageMeter, save_checkpoint

best_acc1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
args = parser.parse_args()


def main():
    global args, best_acc1, device

    criterion = PrototypicalLoss().to(device)

    cudnn.benchmark = True

    runs_path = glob('runs/*')
    max_len_exp = max([len(x) for x in runs_path])
    print(f"|{'Experiment':^max_len_exp}|{'Loss':^10}|{'ACC'}:^10|")

    for exp in glob('runs/*'):
        checkpoint, args = None, None
        files = glob(exp + '/*')
        for file in files:
            if file.endswith('model_best.pth'):
                checkpoint = torch.load(file)
            elif file.endswith('.json'):
                params = json.load(open(file))
                args = SimpleNamespace(**params)

        if checkpoint is None or args is None:
            print(f"checkpoint and params are not exist in {exp}")
            continue

        test_loader = get_dataloader(args, args.dataset, 'test')

        input_dim = 1 if args.dataset == 'omniglot' else 3
        if args.model == 'protonet':
            model = ProtoNet(input_dim).to(device)
            print("ProtoNet loaded")
        else:
            model = ResNet(input_dim).to(device)
            print("ResNet loaded")

        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc1 = checkpoint['best_acc1']

        loss, acc = test(test_loader, model, criterion)

        print()


@torch.no_grad()
def test(test_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    num_support = args.num_support_val

    # switch to evaluate mode
    model.eval()
    for i, data in enumerate(test_loader):
        input, target = data[0].to(device), data[1].to(device)

        output = model(input)
        loss, acc1 = criterion(output, target, num_support)

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))

    return losses.avg, top1.avg


if __name__ == '__main__':
    max_len_exp = 10
    print(f"|{'Experiment':^16}|{'Loss':^10}|{'ACC':^9}|")
