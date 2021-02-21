import os
import sys
import json
from glob import glob
from types import SimpleNamespace
from pprint import pprint as pp

sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(__file__))))

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from arguments import get_args
from dataloader import get_dataloader
from relationnet import RelationNetwork, Embedding
from utils.common import margin_of_error
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

best_acc1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = get_args()
summary = SummaryWriter()


def main():
    global args, best_acc1, device

    # Init seed
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)

    criterion = torch.nn.MSELoss()
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
                if args.dataset == 'omniglot':
                    args.iteration = 1000
                else:
                    args.iteration = 600

        if checkpoint is None or args is None:
            except_list.append(f"checkpoint and params are not exist in {exp}")
            continue

        if args.dataset == 'miniImageNet':
            in_channel = 3
            feature_dim = 64 * 3 * 3
            test_loader = get_dataloader(args, 'val')
        elif args.dataset == 'omniglot':
            in_channel = 1
            feature_dim = 64
            test_loader = get_dataloader(args, 'test')
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported")

        embedding = Embedding(in_channel).to(device)
        model = RelationNetwork(feature_dim).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        embedding.load_state_dict(checkpoint['embedding_state_dict'])
        best_acc1 = checkpoint['best_acc1']

        loss_list, acc_list = test(test_loader, model, embedding, criterion)

        loss, loss_moe = margin_of_error(loss_list)
        acc, acc_moe = margin_of_error(acc_list)

        loss_string = f'{loss:.3f} {pl_mi} {loss_moe:.3f}'
        acc_string = f'{acc:.3f} {pl_mi} {acc_moe:.3f}'

        print(f"|{exp:^{max_len_exp}}|{loss_string:^16}|{acc_string:^16}|")

    pp(except_list)


@torch.no_grad()
def test(test_loader, model, embedding, criterion):
    losses = []
    accuracies = []

    num_class = args.classes_per_it_val
    num_support = args.num_support_val
    num_query = args.num_query_val

    model.eval()
    embedding.eval()
    for i, data in enumerate(test_loader):
        x, y = data[0].to(device), data[1].to(device)
        x_support, x_query, y_query = split_support_query_set(x, y, num_class, num_support, num_query)

        support_vector = embedding(x_support)
        query_vector = embedding(x_query)

        _size = support_vector.size()

        support_vector = support_vector.view(num_class, num_support, _size[1], _size[2], _size[3]).sum(dim=1)
        support_vector = support_vector.repeat(num_class * num_query, 1, 1, 1)
        query_vector = torch.stack([x for x in query_vector for _ in range(num_class)])

        _concat = torch.cat((support_vector, query_vector), dim=1)

        y_pred = model(_concat).view(-1, num_class)

        y_one_hot = torch.zeros(num_query * num_class, num_class).to(device).scatter_(1, y_query.unsqueeze(1), 1)
        loss = criterion(y_pred, y_one_hot)

        losses.append(loss.item())

        y_hat = y_pred.argmax(1)
        accuracy = y_hat.eq(y_query).float().mean()
        accuracies.append(accuracy)

        probs = [el[i].item() for i, el in zip(y_query, y_pred)]

        y_support = torch.LongTensor([x for x in range(num_class) for _ in range(num_support)])

        summary.add_figure('predictions vs. actuals',
                           plot_classes_preds(y_hat, probs, [x_support, x_query],
                                              [y_support, y_query], num_class, num_support, num_query),
                           global_step=i + 1)

        break

    return losses, accuracies


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    img = img.to('cpu')
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(preds, probs, images, labels, N_way, K_shot, K_shot_test):
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
    for row in np.arange(K_shot_test):
        for col in np.arange(N_way):
            ax = fig.add_subplot(2 * K_shot_test, N_way, N_way * K_shot_test + row * N_way + col + 1, xticks=[],
                                 yticks=[])
            matplotlib_imshow(query_images[col * K_shot_test + row], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                preds[col * K_shot_test + row],
                probs[col * K_shot_test + row] * 100.0,
                labels[1][col * K_shot_test + row]),
                color=(
                    "green" if preds[col * K_shot_test + row] == labels[1][col * K_shot_test + row].item() else "red"))
    return fig


def split_support_query_set(x, y, num_class, num_support, num_query):
    num_sample_support = num_class * num_support
    x_support, x_query = x[:num_sample_support], x[num_sample_support:]
    y_support, y_query = y[:num_sample_support], y[num_sample_support:]

    _classes = torch.unique(y_support)

    support_idx = torch.stack(list(map(lambda c: y_support.eq(c).nonzero().squeeze(1), _classes)))
    xs = torch.cat([x_support[idx_list] for idx_list in support_idx])

    query_idx = torch.stack(list(map(lambda c: y_query.eq(c).nonzero().squeeze(1), _classes)))
    xq = torch.cat([x_query[idx_list] for idx_list in query_idx])

    yq = torch.LongTensor([x for x in range(len(_classes)) for _ in range(num_query)]).to(device)

    return xs, xq, yq


if __name__ == '__main__':
    main()
