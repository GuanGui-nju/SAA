import argparse
import os
import yaml


def save_config_yaml(var_dict, file_path):
    if "device" in var_dict.keys():
        # del var_dict["device"]
        var_dict.pop('device')
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(var_dict, f, default_flow_style=False, encoding='utf-8', allow_unicode=True)


def yaml_config_hook(config_file):
    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def create_parser_from_file(name, filepath="./lassl.yaml"):
    parser = argparse.ArgumentParser(description=name)
    config = yaml_config_hook(filepath)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser


def create_parser(name):
    parser = argparse.ArgumentParser(description=name)
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--save_name', type=str, default='daSSL')
    parser.add_argument('--exp_name', type=str, default="baseline")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    
    '''
    Training Configuration of FixMatch
    '''
    
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2**20, 
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10000,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    
    parser.add_argument('--soft_label', action='store_true', help="use soft label")
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    
    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', action='store_true', help='use mixed precision training or not')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', action='store_true', help="use network name")
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    '''
    Data Configurations
    '''
    
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    
    '''
    multi-GPUs & Distrbitued Training
    '''
    
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    
    return parser


def parse_commandline_args(name="dassl", filepath="./lassl.yaml"):
    '''
        args = parser.parse_args('')  # running in ipynb
        args = parser.parse_args()  # running in command line
    '''
    if filepath is not None and os.path.exists(filepath) and not os.path.isdir(filepath):
        args = create_parser_from_file(name, filepath).parse_args()
    else:
        args = create_parser(name).parse_args()
    return args


if __name__ == '__main__':
    test_parse_file = True
    if test_parse_file:
        my_args = parse_commandline_args()
        print("=========parse from file=======")
    else:
        my_args = parse_commandline_args(filepath=None)
        print("=========argparse CLI=======")

    # show info
    for key in my_args.__dict__:
        var = my_args.__dict__[key]
        v_type = type(var)
        print(f"{key}:{var}:{v_type}")
