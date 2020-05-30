import json
from statistics import mean
import numpy as np
import os
from bokeh.plotting import output_file, figure, save
from bokeh.layouts import gridplot
from src.utils.tools import listdir, hash_append


def combined(ids, name, legend=None):
    summaries = {}
    for key_ in [key_ for key_ in settings if len(settings[key_]) > 1]:
        id = [f.split(' := ') for f in key_.split('\n') if f.split(' := ')[0] == 'id'][0][1]
        if not any([id == id_ for id_ in ids]):
            continue
        rewards = {}
        for res_folder in settings[key_]:
            with open(os.path.join(res_folder, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
            for episode, score in zip(metrics['episodes'], metrics['t_scores']):
                if score is not None and episode % 25 == 0:
                    hash_append(rewards, episode, score)
        episodes = sorted(rewards.keys())
        quart1 = [np.percentile(rewards[ep], 25) for ep in episodes]
        median = [np.percentile(rewards[ep], 50) for ep in episodes]
        quart3 = [np.percentile(rewards[ep], 75) for ep in episodes]
        summaries[id] = (quart1, median, quart3)

    episodes = episodes[:min([len(summaries[id][1]) for id in summaries.keys()])]

    COLORS = ("royalblue", "orchid", "sienna", "seagreen")
    output_file(os.path.join('./res/plots', name.lower() + '.html'), title=name)
    s = figure(width=720, height=360, title="Performance", x_axis_label='episodes', y_axis_label='reward',
               y_range=(0, 1000))
    for id in sorted(summaries.keys(), key=lambda x: ids.index(x)):
        ii = ids.index(id)
        s.line(episodes, summaries[id][1][:len(episodes)], line_color=COLORS[ii], line_width=2, line_alpha=0.75,
               legend_label=legend[ids.index(id)])
        s.varea(episodes, summaries[id][0][:len(episodes)], summaries[id][2][:len(episodes)], fill_color=COLORS[ii],
                fill_alpha=0.25)
    s.legend.location = "bottom_right"
    save(gridplot([[s]]))


settings = {}

for results_folder in listdir('./res/results'):
    with open(os.path.join(results_folder, 'hyperparameters.txt'), 'r') as f:
        s = ""
        for line in f.readlines():
            if 'seed' not in line:
                s += line
    try:
        with open(os.path.join(results_folder, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        if len(metrics['steps']) >= 1000:
            hash_append(settings, s, results_folder)
    except FileNotFoundError:
        pass

# for key in [key for key in settings if len(settings[key]) > 1]:
#     id = [f.split(' := ') for f in key.split('\n') if f.split(' := ')[0] == 'id'][0][1]
#     print('ID:', id)
#     rewards = {}
#     for res_folder in settings[key]:
#         with open(os.path.join(res_folder, 'metrics.json'), 'r') as f:
#             metrics = json.load(f)
#         for episode, score in zip(metrics['episodes'], metrics['t_scores']):
#             if score is not None and episode % 25 == 0:
#                 hash_append(rewards, episode, score)
#     episodes = sorted(rewards.keys())
#     quart1 = [np.percentile(rewards[ep], 25) for ep in episodes]
#     median = [np.percentile(rewards[ep], 50) for ep in episodes]
#     quart3 = [np.percentile(rewards[ep], 75) for ep in episodes]
#     # summaries[id] = (quart1, median, quart3)
#
#     output_file(os.path.join('./res/plots', id + '.html'), title=id)
#     s = figure(width=720, height=360, title="Performance", x_axis_label='episodes', y_axis_label='reward',
#                y_range=(0, 800))
#     s.line(episodes, median)
#
#     s.varea(episodes, quart1, quart3, fill_alpha=0.25)
#     save(gridplot([[s]]))
#     pass
#
# pass

# combined(['lov', 'vanilla'], 'validate', legend=['with overshooting', 'without overshooting'])
combined(['gradient', 'latency', 'latency2', 'latency4'], 'latency_combined',
         legend=['no latency', '1 timestep', '2 timesteps', '4 timesteps'])
combined(['gradient', 'ar4', 'ar8', 'ar12'], 'control_frequency',
         legend=['normal', '2x slower', '4x slower', '6x slower'])
# combined()
