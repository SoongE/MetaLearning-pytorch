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

best_acc1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = get_args()


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

        if args.dataset == 'miniImagenet':
            in_channel = 3
            feature_dim = 64 * 3 * 3
            test_loader = get_dataloader(args, 'val')
        elif args.dataset == 'omniglot':
            in_channel = 3
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

    return losses, accuracies


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
