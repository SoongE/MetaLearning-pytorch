import os
import sys
import json
import argparse
from glob import glob
from pprint import pprint as pp
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(__file__))))

import torch
import torch.backends.cudnn as cudnn

from dataloader import get_dataloader
from models.protonet import ProtoNet
from models.resnet import ResNet
from prototypical_loss import PrototypicalLoss

from utils.common import margin_of_error

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
    print(f"|{'Experiment':^{max_len_exp}}|{'Loss':^16}|{'ACC':^16}|")

    except_list = []
    pl_mi = u"\u00B1"

    for exp in glob('runs/*'):
        checkpoint, args = None, None
        files = glob(exp + '/*')
        for file in files:
            if file.endswith('model_best.pth'):
                checkpoint = torch.load(os.path.abspath(file))
            elif file.endswith('.json'):
                params = json.load(open(os.path.abspath(file)))
                args = SimpleNamespace(**params)

        if checkpoint is None or args is None:
            except_list.append(f"checkpoint and params are not exist in {exp}")
            continue

        if args.dataset == 'omniglot':
            args.iteration = 1000
            test_loader = get_dataloader(args, 'test')
        else:
            args.iteration = 600
            test_loader = get_dataloader(args, 'val')

        input_dim = 1 if args.dataset == 'omniglot' else 3
        if args.model == 'protonet':
            model = ProtoNet(input_dim).to(device)
        else:
            model = ResNet(input_dim).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc1 = checkpoint['best_acc1']

        loss_list, acc_list = test(test_loader, model, criterion)

        loss, loss_moe = margin_of_error(loss_list)
        acc, acc_moe = margin_of_error(acc_list)

        loss_string = f'{loss:.3f} {pl_mi} {loss_moe:.3f}'
        acc_string = f'{acc:.3f} {pl_mi} {acc_moe:.3f}'

        print(f"|{exp:^{max_len_exp}}|{loss_string:^16}|{acc_string:^16}|")

    if len(except_list):
        pp(except_list)


@torch.no_grad()
def test(test_loader, model, criterion):
    num_support = args.num_support_val
    losses = []
    top1 = []
    # switch to evaluate mode
    model.eval()

    for i, data in enumerate(test_loader):
        input, target = data[0].to(device), data[1].to(device)

        output = model(input)
        loss, acc1 = criterion(output, target, num_support)

        losses.append(loss)
        top1.append(acc1)

    return losses, top1


if __name__ == '__main__':
    main()
