import os
import torch
import numpy as np
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res
    # return res, pred[:1].squeeze(0)


def save_checkpoint(state, is_best, args):
    directory = args.log_dir
    filename = directory + f"/checkpoint_{state['epoch']}.pth"

    if not os.path.exists(directory):
        os.makedirs(directory)

    if is_best:
        filename = directory + "/model_best.pth"
        torch.save(state, filename)
    else:
        torch.save(state, filename)


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
        contain prediction probability range from 0.0 ~ 1.0 formed softmax.
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
    probs = [el[i].item() for i, el in zip(labels[1], probs)]

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
