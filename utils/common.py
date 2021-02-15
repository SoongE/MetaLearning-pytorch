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
