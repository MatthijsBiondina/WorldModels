import json
from statistics import mean
import numpy as np
import os
from bokeh.plotting import output_file, figure, save
from bokeh.layouts import gridplot
from src.utils.tools import listdir, hash_append


def combined(ids, name, legend=None, y_range=(0, 900)):
    summaries = {}
    episodes = []
    for key_ in settings:  # [key_ for key_ in settings if len(settings[key_]) > 1]:
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
        episodes = episodes[:min(39, len(episodes))]
        quart1 = [np.percentile(rewards[ep], 25) for ep in episodes]
        median = [np.percentile(rewards[ep], 50) for ep in episodes]
        quart3 = [np.percentile(rewards[ep], 75) for ep in episodes]
        summaries[id] = (quart1, median, quart3)
    COLORS = ("royalblue", "orchid", "seagreen", "sienna", "darkkhaki")
    output_file(os.path.join('./res/plots', name.lower() + '.html'), title=name)
    s = figure(width=720, height=int(360 * (y_range[1] - y_range[0]) / 900), title="Performance",
               x_axis_label='episodes', y_axis_label='score', y_range=y_range)
    for id in sorted(summaries.keys(), key=lambda x: ids.index(x)):
        ii = ids.index(id)
        s.line(episodes[:len(summaries[id][1])], summaries[id][1], line_color=COLORS[ii], line_width=2, line_alpha=0.75,
               legend_label=legend[ids.index(id)])
        s.varea(episodes[:len(summaries[id][1])], summaries[id][0], summaries[id][2], fill_color=COLORS[ii],
                fill_alpha=0.25)
    # s.legend.location = "top_left"
    s.legend.location = "bottom_right"
    save(gridplot([[s]]))


settings = {}
folders = []
for fname in ('gradient', 'imitation', 'planet'):
    for folder in listdir(f"../{fname}/res/results"):
        folders.append((folder, fname))

for results_folder, algorithm in folders:
    with open(os.path.join(results_folder, 'hyperparameters.txt'), 'r') as f:
        s = ""
        for line in f.readlines():
            if 'seed' not in line:
                if line.split(' ')[0] == 'id':
                    id = f"id := {algorithm}_{line.split(' ')[-1]}"
                    print(id.replace('\n', ''))
                    s += id
                else:
                    s += line
    try:
        with open(os.path.join(results_folder, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        if len(metrics['steps']) >= 1000 or 'data_aggregation' in id:
            hash_append(settings, s, results_folder)
    except FileNotFoundError:
        pass

combined(['planet_lov', 'planet_vanilla'], 'planet_validation',
         legend=['With Latent Overshooting', 'Without Latent Overshooting'])
combined(['planet_lov', 'gradient_ar4'], 'planet_gradient',
         legend=['CEM Planner', 'Gradient-Based Optimization'])
combined(['planet_lov', 'imitation_data_aggregation', 'imitation_policy_aggregation'], 'planet_imitation',
         legend=['CEM Planner', 'Data Aggregation', 'Policy Aggregation'], y_range=(-200, 900))
combined(['planet_lov', 'planet_latency', 'planet_latency2', 'planet_latency4'], 'latency_planet',
         legend=['no latency', '1 timestep', '2 timesteps', '4 timesteps'])
combined(['gradient_ar4', 'gradient_ar4_lat1', 'gradient_ar4_lat2', 'gradient_ar4_lat4', 'gradient_ar4_lat8'], 'latency_gradient',
         legend=['no latency', '1 timestep', '2 timesteps', '4 timesteps', '8 timesteps'])

combined(['planet_lov', 'planet_ar4', 'planet_ar8', 'planet_ar12'], 'planet_cf',
         legend=['2 timesteps', '4 timesteps', '8 timesteps', '12 timesteps'])
combined(['gradient_ar2_4andreas', 'gradient_ar4', 'gradient_ar8', 'gradient_ar12'], 'gradient_cf',
         legend=['2 timesteps', '4 timesteps', '8 timesteps', '12 timesteps'])
