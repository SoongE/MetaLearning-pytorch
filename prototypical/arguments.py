import argparse

from utils.common import CustomArgs

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp_name',
                    type=str,
                    help='experiment name',
                    default='TEST')

parser.add_argument('-epo', '--epochs',
                    type=int,
                    help='number of epochs to train for',
                    default=100)

parser.add_argument('--lr', '--learning_rate',
                    type=float,
                    help='learning rate for the model, default=0.001',
                    default=0.001)

parser.add_argument('-lrS', '--lr_scheduler_step',
                    type=int,
                    help='StepLR learning rate scheduler step, default=20',
                    default=20)

parser.add_argument('-lrG', '--lr_scheduler_gamma',
                    type=float,
                    help='StepLR learning rate scheduler gamma, default=0.5',
                    default=0.5)

parser.add_argument('-its', '--iterations',
                    type=int,
                    help='number of episodes per epoch, default=100',
                    default=100)

parser.add_argument('-cTr', '--classes_per_it_tr',
                    type=int,
                    help='number of random classes per episode for training, default=60',
                    default=60)

parser.add_argument('-nsTr', '--num_support_tr',
                    type=int,
                    help='number of samples per class to use as support for training, default=5',
                    default=5)

parser.add_argument('-nqTr', '--num_query_tr',
                    type=int,
                    help='number of samples per class to use as query for training, default=5',
                    default=5)

parser.add_argument('-cVa', '--classes_per_it_val',
                    type=int,
                    help='number of random classes per episode for validation, default=5',
                    default=5)

parser.add_argument('-nsVa', '--num_support_val',
                    type=int,
                    help='number of samples per class to use as support for validation, default=5',
                    default=5)

parser.add_argument('-nqVa', '--num_query_val',
                    type=int,
                    help='number of samples per class to use as query for validation, default=15',
                    default=15)

parser.add_argument('-seed', '--manual_seed',
                    type=int,
                    help='input for the manual seeds initializations',
                    default=7)

parser.add_argument('--test_iter',
                    type=int,
                    help='number of epochs to train, default = 50',
                    default=20)

parser.add_argument('--log_dir',
                    default='runs',
                    type=str,
                    help='root where to store models, losses and accuracies')

parser.add_argument('-d', '--dataset',
                    type=str,
                    help="Select dataset [omniglot | miniImageNet]",
                    default='omniglot')

parser.add_argument('-m', '--model',
                    type=str,
                    help='Select model [protonet | resnet]',
                    default='protonet')

parser.add_argument('--resume', action="store_true", help="resume train")

parser.set_defaults(resume=False)


def get_args():
    return CustomArgs(parser).get()
