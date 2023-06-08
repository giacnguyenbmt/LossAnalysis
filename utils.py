import os
import yaml

import torch


def load_config(config_path):
    _, ext = os.path.splitext(config_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def parse_opt(opts):
    config = {}
    if not opts:
        return config
    for s in opts:
        s = s.strip()
        k, v = s.split('=')
        config[k] = yaml.load(v, Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def parse_config(args):
    config = load_config(args.config)
    if config['Global']['blank_first']:
        config['Global']['blank_idx'] = 0
        config['Global']['chars'] = ' ' + config['Global']['chars']
    else:
        config['Global']['blank_idx'] = len(config['Global']['chars'])
        config['Global']['chars'] = config['Global']['chars'] + ' '

    if args.__dict__.get('pretrained_model', None) is not None:
        config['Global']['pretrained_model'] = args.pretrained_model

    config['Architecture']['class_num'] = len(config['Global']['chars'])
    config = merge_config(config, parse_opt(args.opt))
    return config


def load_pretrained_model(model, pretrained_model_path, device):
    if device == torch.device("cpu"):
        state_dict = torch.load(pretrained_model_path, map_location=device)
    else:
        state_dict = torch.load(pretrained_model_path)
    filtered_state_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if state_dict.get(key, None) is not None:
            if state_dict[key].shape == model_dict[key].shape:
                filtered_state_dict[key] = state_dict[key]
    model.load_state_dict(filtered_state_dict, strict=True)
    return model
