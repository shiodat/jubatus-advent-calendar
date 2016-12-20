#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from jubakit.classifier import Classifier, Schema, Dataset, Config
from jubakit.model import JubaDump
from sklearn.datasets import make_classification
from sklearn.utils import shuffle, check_random_state
from sklearn.preprocessing import LabelEncoder, StandardScaler


# user parameters
port = 9299
method = sys.argv[1]
nearest_neighbor_num = 5
local_sensitivity = 1
hash_num = 512
seed = 42
meshsize = 50   # we can draw clear decision surface with large meshsize

# setting random seed
np.random.seed(seed)
check_random_state(seed)

# load dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1, n_classes=3, flip_y=0)
labels = np.array(['c1', 'c2', 'c3'])
y = labels[y]
X, y = shuffle(X, y, random_state=42)        # sklearn iris dataset is unshuffled

# prepare encoder to plot decision surface
le = LabelEncoder()
le.fit(labels)
c = le.transform(y)

# scale dataset with (mean, variance) = (0, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# calculate the domain
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X0, X1 = np.meshgrid(np.linspace(X_min[0], X_max[0], meshsize),
                     np.linspace(X_min[1], X_max[1], meshsize))

# make training dataset
dataset = Dataset.from_array(X, y)
# make mesh dataset to plot decision surface
contourf_dataset = Dataset.from_array(np.c_[X0.ravel(), X1.ravel()])

# setup and run jubatus
config = Config(method=method,
                parameter={
                    'nearest_neighbor_num': nearest_neighbor_num,
                    'local_sensitivity': local_sensitivity})
classifier = Classifier.run(config, port=port)

# construct classifier prediction models and dump model weights
for i, _ in enumerate(classifier.train(dataset)):
    model_name = 'nn_decision_surface_{}'.format(i)
    classifier.save(name=model_name)

# prepare figure
fig, ax = plt.subplots()

def draw_decision_surface(i):
    midx = int(i / 2)
    sidx = int(i / 2) + (i % 2)
    # load jubatus prediction model
    model_name = 'nn_decision_surface_{}'.format(midx)
    classifier.load(name=model_name)

    # predict 
    Y_pred = []
    for (_, _, result) in classifier.classify(contourf_dataset):
        y_pred = le.transform(result[0][0])
        Y_pred.append(y_pred)
    Y_pred = np.array(Y_pred).reshape(X0.shape)  

    # draw decision surface
    ax.clear()
    ax.set_xlim([X_min[0], X_max[0]])
    ax.set_ylim([X_min[1], X_max[1]])
    ax.contourf(X0, X1, Y_pred, alpha=0.3, cmap=plt.cm.jet)
    ax.scatter(X[:sidx+1][:, 0], X[:sidx+1][:, 1], c=c[:sidx+1], s=60, cmap=plt.cm.jet)
    ax.set_title('method={}, iteration={}'.format(method, sidx))
    return ax

ani = FuncAnimation(fig, draw_decision_surface, frames=np.arange(0, X.shape[0]*2), interval=100)
ani.save('{}.gif'.format(method), writer='imagemagick')
plt.show()
classifier.stop()