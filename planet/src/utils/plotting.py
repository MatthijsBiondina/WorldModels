import json

import os
from bokeh.plotting import output_file, figure, save
from bokeh.layouts import gridplot
import src.utils.config as cfg


def save_metrics(metrics: dict, save_loc: str):
    with open(os.path.join(save_loc, 'metrics.json'), 'w+') as f:
        json.dump(metrics, f, indent=2)

    output_file(os.path.join(save_loc, 'plt.html'), title=save_loc.split('/')[-1])

    s_top = figure(width=720, height=360, title="Performance", x_axis_label='episodes', y_axis_label='reward')
    s_top.line(not_none(metrics['episodes'], metrics['rewards']), not_none(metrics['rewards']),
               legend_label="With Action Noise", line_color="orchid", line_width=3, line_alpha=0.66)
    s_top.line(not_none(metrics['episodes'], metrics['t_scores']), not_none(metrics['t_scores']),
               legend_label="Without Action Noise", line_color="royalblue", line_width=3, line_alpha=0.66)
    s_top.legend.location = "bottom_right"

    s_bot = figure(width=720, height=360, x_range=s_top.x_range, title="Loss Scores",
                   x_axis_label="episode", y_axis_label='loss')
    s_bot.line(not_none(metrics['episodes'], metrics['o_loss']), not_none(metrics['o_loss']),
               legend_label="Observation Loss (MSE)", line_color="orchid", line_width=3, line_alpha=0.66)
    s_bot.line(not_none(metrics['episodes'], metrics['r_loss']),
               list(map(lambda x: x / cfg.action_repeat, not_none(metrics['r_loss']))),
               legend_label="Reward Loss (MSE)", line_color="royalblue", line_width=3, line_alpha=0.66)
    s_bot.line(not_none(metrics['episodes'], metrics['kl_loss']),
               list(map(lambda x: x / (1 + cfg.overshooting_kl_beta) - cfg.free_nats, not_none(metrics['kl_loss']))),
               legend_label="Complexity Loss (KL-divergence)", line_color="sienna", line_width=3, line_alpha=0.66)
    # s_bot.line(not_none(metrics['episodes'], metrics['p_loss']),
    #            list(map(lambda x: x / cfg.action_repeat, not_none(metrics['p_loss']))),
    #            legend_label="Policy Loss (MSE)", line_color="seagreen", line_width=3)

    p = gridplot([[s_top], [s_bot]])

    save(p)

    pass


def not_none(vlist, klist=None):
    if klist is None:
        return [x for x in vlist if x is not None]
    else:
        return [x for x, k in zip(vlist, klist) if k is not None]
