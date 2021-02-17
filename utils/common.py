import os
import json
import math


class CustomArgs:
    def __init__(self, parser):
        super().__init__()
        args = parser.parse_args()
        args.log_dir = os.path.join(args.log_dir, args.exp_name)

        self.args = args

        if args.resume:
            self.load_args()
        else:
            self.save_args()

    def save_args(self):
        directory = self.args.log_dir
        param_path = os.path.join(directory, 'params.json')

        if not os.path.exists(f'runs/{self.args.exp_name}'):
            os.makedirs(directory)

        if not os.path.isfile(param_path):
            print(f"Save params in {param_path}")

            all_params = self.args.__dict__
            with open(param_path, 'w') as fp:
                json.dump(all_params, fp, indent=4, sort_keys=True)
        else:
            print(f"Config file already exist.")
            # raise ValueError

    def load_args(self):
        param_path = os.path.join(self.args.log_dir, 'params.json')
        params = json.load(open(param_path))

        self.args.__dict__.update(params)

        self.args.resume = True

    def get(self):
        return self.args


def margin_of_error(values, confidence_interval=1.96):
    num = len(values)
    mean = sum(values) / num
    variance = sum(list(map(lambda x: pow(x - mean, 2), values))) / num

    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(num)

    return mean, standard_error * confidence_interval


# from https://github.com/oscarknagg/few-shot
def pairwise_distances(x, y, matching_fn):
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn.lower() == 'l2' or matching_fn.lower == 'euclidean':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn.lower() == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn.lower() == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))
