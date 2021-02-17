import os
import sys
from glob import glob

sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(__file__))))

import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from arguments import get_args
from dataloader import get_dataloader
from matchnet import MatchingNetworks
from utils.train_utils import AverageMeter, save_checkpoint

best_acc1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = get_args()
writer = SummaryWriter(args.log_dir)


def main():
    global args, best_acc1, device

    # Init seed
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)

    if args.dataset.lower() == 'miniimagenet':
        train_loader, val_loader = get_dataloader(args, 'train', 'val')
        in_channel = 3
        lstm_input_size = 1600
    elif args.dataset.lower() == 'omniglot':
        train_loader, val_loader = get_dataloader(args, 'trainval', 'test')
        in_channel = 1
        lstm_input_size = 64
    else:
        raise KeyError(f"Dataset {args.dataset} is not supported")

    model = MatchingNetworks(args.classes_per_it_tr, args.num_support_tr, args.num_query_tr, args.num_query_val,
                             in_channel, args.lstm_layers, lstm_input_size, args.unrolling_steps, fce=True,
                             distance_fn='cosine').to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    cudnn.benchmark = True

    if args.resume:
        try:
            checkpoint = torch.load(sorted(glob(f'{args.log_dir}/checkpoint_*.pth'), key=len)[-1])
        except:
            checkpoint = torch.load(args.log_dir + '/model_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

        print(f"load checkpoint {args.exp_name}")
    else:
        start_epoch = 0

    print(f"model parameter : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(train_loader, model, optimizer, criterion, epoch)
        val_loss, acc1 = validate(val_loader, model, criterion, epoch)

        if acc1 >= best_acc1:
            is_best = True
            best_acc1 = acc1
        else:
            is_best = False

        if epoch % args.save_iter == 0 or is_best:
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc1': best_acc1,
                'epoch': epoch,
            }, is_best, args)

            writer.add_scalar("Acc/BestAcc", acc1, epoch)

        print(f"[{epoch}/{args.epochs}] {train_loss:.3f}, {val_loss:.3f}, {acc1:.3f}, # {best_acc1:.3f}")

    writer.close()


def train(train_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    total_epoch = len(train_loader) * epoch

    model.train()
    model.custom_train()
    for i, data in enumerate(train_loader):
        x, y = data[0].to(device), data[1].to(device)

        y_pred, y = model(x, y)

        loss = criterion(y_pred, y)

        losses.update(loss.item(), y_pred.size(0))

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        writer.add_scalar("Loss/Train", loss.item(), total_epoch + i)

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    accuracies = AverageMeter()
    total_epoch = len(val_loader) * epoch

    model.eval()
    model.custom_eval()
    for i, data in enumerate(val_loader):
        x, y = data[0].to(device), data[1].to(device)

        y_pred, y = model(x, y, is_train=False)

        loss = criterion(y_pred, y)
        acc = y_pred.argmax(dim=1).eq(y).float().mean()

        losses.update(loss.item(), y_pred.size(0))
        accuracies.update(acc.item(), y_pred.size(0))

        writer.add_scalar("Loss/Val", loss.item(), total_epoch + i)
        writer.add_scalar("Acc/Val", acc.item(), total_epoch + i)

    return losses.avg, accuracies.avg


if __name__ == '__main__':
    main()
