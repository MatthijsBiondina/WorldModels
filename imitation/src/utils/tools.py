import json

import cv2
import os
import sys
import traceback
from datetime import datetime

from torch import nn

import src.utils.config as cfg
from tqdm import tqdm


def init_weights(m):
    try:
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias, 0., 0.1)
    except AttributeError:
        pass


def pyout(*args, ex=None):
    """
    Print with part of trace for debugging.

    :param args: arguments to print
    :param ex: if not None, exits application with provided exit code
    :return:
    """
    trace = traceback.format_stack()[-2].split('\n')
    _tqdm_write("\033[1;33m" + trace[0].split(', ')[0].replace('  ', '') + "\033[0m")
    _tqdm_write("\033[1;33m" + trace[0].split(', ')[1].split(' ')[1] + ':', trace[1].replace('    ', ''), "\033[0m")
    _tqdm_write(*args)
    _tqdm_write("")
    if ex is not None:
        sys.exit(ex)


def make_video(frames, save_loc, epoch):
    os.makedirs(os.path.join(save_loc, 'videos'), exist_ok=True)
    H, W, _ = frames[0].shape
    writer = cv2.VideoWriter(os.path.join(save_loc, 'videos', str(epoch).zfill(4) + '.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 30. / cfg.action_repeat, (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_metrics(metrics: dict, save_loc: str):
    with open(os.path.join(save_loc, 'metrics.json'), 'w+') as f:
        json.dump(metrics, f, indent=2)


def poem(desc: str) -> str:
    """
    Format description for tqdm bar. Assumes desired width of 23 characters for description. Adds whitespace if
    description is shorter, clips if description is longer.

    :param desc: description
    :return: formatted description
    """
    if len(desc) < 23:
        return desc + ' ' * (23 - len(desc))
    else:
        return desc[:20] + '...'


def _tqdm_write(*args):
    tqdm.write(datetime.now().strftime("%d-%m-%Y %H:%M"), end=' ')
    for arg in list(map(str, args)):
        for ii, string in enumerate(arg.split('\n')):
            tqdm.write((' ' * 17 if ii > 0 else '') + string, end=' ' if ii == len(arg.split('\n')) - 1 else '\n')
    tqdm.write('')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def soft_update(target, source, tau):
    """
    update parameters of target network with incremental step towards source network

    Args:
        target: torch.nn.Module - target network
        source: torch.nn.Module - source network
        tau:    float - step size
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)