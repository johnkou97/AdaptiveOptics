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

    def __init__(self, title=None, ylabel="Reward", y_lim=(0, 1), length=100000, figsize=(16, 9)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlabel('Steps', fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=18)
        if y_lim is not None:
            self.ax.set_ylim(y_lim)
        if title is not None:
            self.ax.set_title(title)
        self.ax.set_xlim(0, length)
        # make xticks nice and readable
        self.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        self.ax.tick_params(axis='both', which='major', labelsize=16)
        # make minor ticks 
        self.ax.minorticks_on()
        # set grid that is not too intrusive
        self.ax.grid(which='major', linestyle='--', linewidth='0.5', color='gray', alpha=0.5)
        # make sure the grid is behind the plot
        self.ax.set_axisbelow(True)
        # make background plot background black
        # self.ax.set_facecolor('xkcd:black')


    def add_curve(self, y, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        skip = 1 # only plot every nth point
        x = np.arange(len(y[0]))
        x = x[::skip]
        mean = np.mean(y, axis=0)
        mean = mean[::skip]
        std = np.std(y, axis=0)
        std = std[::skip]
        if label is not None:
            self.ax.plot(x, mean, label=label, linewidth=2)
            self.ax.fill_between(x, (mean - std), (mean + std), alpha=.4)
        else:
            self.ax.plot(x, mean, linewidth=2)
            self.ax.fill_between(x, (mean - std), (mean + std), alpha=.4)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png', legend_pos='best', legend_fontsize=18, legend_down=False, dpi=400):
        ''' name: string for filename of saved figure '''
        if legend_down:
            # Put a legend below current axis
            self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=legend_fontsize,
                           frameon=False)
            self.fig.tight_layout()
        else:
            print('1')
            self.ax.legend(loc=legend_pos, fontsize=legend_fontsize, frameon=True, facecolor='lightgrey')
            self.fig.tight_layout()
        self.fig.savefig(name, dpi=dpi)

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