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
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

best_acc1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
args = parser.parse_args()
summary = SummaryWriter()


def main():
    global args, best_acc1, device

    criterion = PrototypicalLoss().to(device)

    cudnn.benchmark = True

    runs_path = glob('runs/*')
    max_len_exp = max([len(x) for x in runs_path]) + 2
    print(f"|{'Experiment':^{max_len_exp}}|{'Loss':^17}|{'ACC':^17}|")

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
        x, y = data[0].to(device), data[1].to(device)

        y_pred = model(x)
        loss, acc1 = criterion(y_pred, y, num_support)

        losses.append(loss)
        top1.append(acc1)

        print(y_pred)
        print(y)
        # probs = [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(y, y_pred)]
        #
        # summary.add_figure('predictions vs. actuals',
        #                    plot_classes_preds(predict_labels, probs, [sample_images, test_images],
        #                                       [sample_labels, test_labels], [sample_class, test_class]),
        #                    global_step=episode + 1)
        # write_plot_to_tensorboard = False

    return losses, top1


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(preds, probs, images, labels, N_way, K_shot):
    """
    Plot Prediction Samples.
    Parameters
    ----------
    preds : list
        contain prediction value range from 0 to N_way - 1.
    probs : list
        contain prediction probability range from 0.0 ~ 1.0.
    images : list
        images[0] is sample images. Shape is (N_way, K_shot, 1, 28, 28).
        images[1] is query images. Shape is (N_way, K_shot, 1, 28, 28).
    labels: list
        labels[0] contains y value for sample image.
        labels[1] contains y value for query image.
    """
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(30, 20))
    sample_images, query_images = images
    # display sample images
    for row in np.arange(K_shot):
        for col in np.arange(N_way):
            ax = fig.add_subplot(2 * K_shot, N_way, row * N_way + col + 1, xticks=[], yticks=[])
            matplotlib_imshow(sample_images[col * K_shot + row], one_channel=True)
            ax.set_title(labels[0][col * K_shot + row].item())
    # display query images
    for row in np.arange(K_shot):
        for col in np.arange(N_way):
            ax = fig.add_subplot(2 * K_shot, N_way, N_way * K_shot + row * N_way + col + 1, xticks=[], yticks=[])
            matplotlib_imshow(query_images[col * K_shot + row], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                preds[col * K_shot + row],
                probs[col * K_shot + row] * 100.0,
                labels[1][col * K_shot + row]),
                color=("green" if preds[col * K_shot + row] == labels[1][col * K_shot + row].item() else "red"))
    return fig


if __name__ == '__main__':
    main()
