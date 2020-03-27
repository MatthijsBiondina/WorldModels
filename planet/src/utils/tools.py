import json

import cv2
import os
import sys
import traceback
from datetime import datetime
import src.utils.config as cfg
from tqdm import tqdm


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
