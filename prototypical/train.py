import os
import sys
from glob import glob

sys.path.append(os.path.dirname(os.path.realpath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from arguments import get_args
from dataloader import get_dataloader
from models.protonet import ProtoNet
from models.resnet import ResNet
from prototypical_loss import PrototypicalLoss

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

    if args.dataset == 'omniglot':
        train_loader, val_loader = get_dataloader(args, 'trainval', 'test')
        input_dim = 1
    else:
        train_loader, val_loader = get_dataloader(args, 'train', 'val')
        input_dim = 3

    if args.model == 'protonet':
        model = ProtoNet(input_dim).to(device)
        print("ProtoNet loaded")
    else:
        model = ResNet(input_dim).to(device)
        print("ResNet loaded")

    criterion = PrototypicalLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    cudnn.benchmark = True

    if args.resume:
        try:
            checkpoint = torch.load(sorted(glob(f'{args.log_dir}/checkpoint_*.pth'), key=len)[-1])
        except Exception:
            checkpoint = torch.load(args.log_dir + '/model_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

        print(f"load checkpoint {args.exp_name}")
    else:
        start_epoch = 1

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=args.lr_scheduler_gamma,
                                                step_size=args.lr_scheduler_step)

    print(f"model parameter : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(start_epoch, args.epochs + 1):

        train_loss = train(train_loader, model, optimizer, criterion, epoch)

        is_test = False if epoch % args.test_iter else True
        if is_test or epoch == args.epochs or epoch == 1:

            val_loss, acc1 = validate(val_loader, model, criterion, epoch)

            if acc1 >= best_acc1:
                is_best = True
                best_acc1 = acc1
            else:
                is_best = False

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer_state_dict': optimizer.state_dict(),
            }, is_best, args)

            if is_best:
                writer.add_scalar("BestAcc", acc1, epoch)

            print(f"[{epoch}/{args.epochs}] {train_loss:.3f}, {val_loss:.3f}, {acc1:.3f}, # {best_acc1:.3f}")

        else:
            print(f"[{epoch}/{args.epochs}] {train_loss:.3f}")

        scheduler.step()

    writer.close()


def train(train_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    num_support = args.num_support_tr
    total_epoch = len(train_loader) * (epoch - 1)

    # switch to train mode
    model.train()
    for i, data in enumerate(train_loader):
        x, y = data[0].to(device), data[1].to(device)

        y_pred = model(x)
        loss, acc1 = criterion(y_pred, y, num_support)

        losses.update(loss.item(), x.size(0))

        # compute gradient and do optimize step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/Train", loss.item(), total_epoch + i)
        writer.add_scalar("Acc/Train", acc1.item(), total_epoch + i)

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    num_support = args.num_support_val
    total_epoch = len(val_loader) * (epoch - 1)

    # switch to evaluate mode
    model.eval()
    for i, data in enumerate(val_loader):
        x, y = data[0].to(device), data[1].to(device)

        y_pred = model(x)
        loss, acc1 = criterion(y_pred, y, num_support)

        losses.update(loss.item(), x.size(0))
        top1.update(acc1.item(), x.size(0))

        writer.add_scalar("Loss/Val", loss.item(), total_epoch + i)
        writer.add_scalar("Acc/Val", acc1.item(), total_epoch + i)

    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
