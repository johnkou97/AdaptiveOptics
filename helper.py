#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd


class LearningCurvePlot:

    def __init__(self, title=None, ylabel="Reward", y_lim=(-10, 37)):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel(ylabel)
        if y_lim is not None:
            self.ax.set_ylim(y_lim)
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)
        if label is not None:
            self.ax.plot(mean, label=label)
            self.ax.fill_between(list(range(len(mean))), (mean - std), (mean + std), alpha=.2)
        else:
            self.ax.plot(mean)
            self.ax.fill_between(list(range(len(mean))), (mean - std), (mean + std), alpha=.2)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png', legend_pos='best', legend_fontsize='medium', legend_down=False):
        ''' name: string for filename of saved figure '''
        if legend_down:
            # Put a legend below current axis
            self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=legend_fontsize,
                           frameon=False)
            self.fig.tight_layout()
        else:
            self.ax.legend(loc=legend_pos, fontsize=legend_fontsize)
        self.fig.savefig(name, dpi=300)


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)


def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp  # scale by temperature
    z = x - max(x)  # substract max to prevent overflow of softmax
    return np.exp(z) / np.sum(np.exp(z))  # compute softmax


def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)


def linear_anneal(t, T, start, final, percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    '''
    final_from_T = int(percentage * T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t) / final_from_T