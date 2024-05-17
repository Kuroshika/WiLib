import argparse
import os
import colorama
import shutil
import yaml
from easydict import EasyDict as ed
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
        return v
    else:
        return not bool(v in ('f', 'F', 'False', 'false'))


def set_outpath(args, block, root='./outputs'):
    if args.output_path is None:
        path_str = os.path.join(root)
        if args.debug:
            args.output_path = os.path.join(path_str, "debug")
        else:
            dataset_name = args.data_param['type']
            path_str = os.path.join(path_str, dataset_name)
            if dataset_name == 'MMFi':
                protocol = args.data_param['protocol']
                path_str = os.path.join(path_str, protocol)
                split = args.data_param['split_to_use'].replace("_split", "")
                path_str = os.path.join(path_str, split)
            current_time = datetime.now()
            model_name = args.model_param['type']
            path_str = os.path.join(path_str, model_name)
            formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
            path_str = os.path.join(path_str, f"[{formatted_time}]{model_name}")
            args.output_path = path_str


def parser_args(block):
    # params
    parser = argparse.ArgumentParser()

    parser.add_argument('-run_mode', default='train_val')

    # This is a path to the config file which could cover the default args
    parser.add_argument('-config', default='./config/ut_that.yaml')

    # set this symbal False to disable the debug mode
    # F, f, False, false all can work well which is defined in function str2bool() above.
    parser.add_argument('-debug', type=str2bool, default=False)

    # These params are related to the builder which could build module from these param config
    parser.add_argument('-loss', default='cross_entropy')
    parser.add_argument('-model_param', default={})
    parser.add_argument('-head_param', default={})
    parser.add_argument('-training_param', default={})
    parser.add_argument('-optimizer_param', default='sgd_nev')
    parser.add_argument('-data_param', default={})

    # param about devices
    parser.add_argument('-device_id', default=[0])
    parser.add_argument('-cuda_visible_device', default='0')

    # output path to the exact dir, also you can set it None to let the code generate a default path
    parser.add_argument('-output_path', default=None)

    # load checkpoint from here
    parser.add_argument('-pretrained_model', default=None)

    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_device

    if args.debug:
        os.environ['DISPLAY'] = 'localhost:10.0'

    set_outpath(args, block)

    if os.path.isdir(args.output_path) and not args.debug:
        print('log_dir: ' + args.output_path + ' already exist')
        answer = input('delete it? y/n:')
        if answer == 'y':
            shutil.rmtree(args.output_path)
            print('Dir removed: ' + args.output_path)
            input('refresh it')
        else:
            print('Dir not removed: ' + args.output_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    block.addr = os.path.join(args.output_path, 'log.txt')
    block.out_path = os.path.join(args.output_path)
    block.log(f"save model at {args.output_path}")
    parser.add_argument('--IGNORE', action='store_true')
    # 会返回列表
    defaults = vars(parser.parse_args(['--IGNORE']))

    for argument, value in sorted(vars(args).items()):
        reset = colorama.Style.RESET_ALL
        color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
        block.log('{}{}: {}{}'.format(color, argument, value, reset))

    block.copy2out(__file__)
    block.copy2out(args.config)
    args = ed(vars(args))
    return args
