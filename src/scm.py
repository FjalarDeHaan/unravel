#!/bin/env python3
#
# scm.py - sandpit to explore causal models and causal discovery.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2022-08-18
# Last modified: 2022-08-18
#

import cdt
import dowhy
import dowhy.gcm
from dowhy.gcm import EmpiricalDistribution, AdditiveNoiseModel
from dowhy.gcm.ml import create_linear_regressor

import networkx as nx

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, uniform, expon

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

glasso = cdt.independence.graph.Glasso()
ges = cdt.causality.graph.GES()
pc = cdt.causality.graph.PC()

# ################################### #
# Generate some simple 'causal' data. #
# ################################### #

# Sample size N.
N = 1000
# Two exponential random variables X1, X2.
X_1 = expon.rvs(size=N)
X_2 = expon.rvs(size=N, scale=10)
# A 'causally' dependent variable Y.
a_0, a_1, a_2 = 3, 4, 5
Y = a_0 + a_1 * X_1 + a_2 * X_2
# A variable measuring Y with error.
E = Y.min() * norm.rvs(size=N)
Z = Y + E
data = pd.DataFrame({'X_1': X_1, 'X_2': X_2, 'Y': Y, 'Z': Z})

# Create a causal graph using the GES algorithm from the CDT library.
# g = ges.predict(data)
# g = pc.predict(data)
g = nx.read_gml('simple.gml')

# Set up the causal-inference model with the DoWhy library tools.
csm = dowhy.gcm.StructuralCausalModel(g)
csm.set_causal_mechanism('X_1', EmpiricalDistribution())
csm.set_causal_mechanism('X_2', EmpiricalDistribution())
csm.set_causal_mechanism('Y', AdditiveNoiseModel(create_linear_regressor()))
csm.set_causal_mechanism('Z', AdditiveNoiseModel(create_linear_regressor()))

dowhy.gcm.fit(csm, data)

interv1 = {'X_1': lambda x_1: 100}
interv2 = {'X_2': lambda x_2: 1000}

samples = dowhy.gcm.interventional_samples(csm, {'X_1': lambda x_1: 100}, num_samples_to_draw=1000)

smpls = dowhy.gcm.interventional_samples(csm, interv2, num_samples_to_draw=1000)

def badag(n, m):
    """Generate a 'random' digraph based on a BA graph."""
    # Start from a BA graph (undirected).
    ba = nx.barabasi_albert_graph(n, m)
    # Break cycles by removing an edge in each.
    # TODO: Improve this by choosing a random edge in each cycle.
    edges_to_erase = {(cycle[0], cycle[1]) for cycle in nx.cycle_basis(ba)}
    ba.remove_edges_from(edges_to_erase)
    # Turn the graph into a directed graph (each edge becomes two arcs).
    diba = ba.to_directed()
    # Remove one of those arcs at random.
    arcs_to_erase = []
    for edge in diba.edges:
        if edge[0] < edge[1]: # Consider each edge only once.
            toss = np.random.randint(2) # Flip a coin.
            if toss == 0: # Remove either the one arc...
                arcs_to_erase.append(edge)
            else: # Or the other...
                arc_to_erase = edge[1], edge[0]
                arcs_to_erase.append(arc_to_erase)
    diba.remove_edges_from(arcs_to_erase)
    return diba

def N(t, r=1, K=1, N_0=.001):
    denominator = 1 + ((K - N_0)/N_0)*np.exp(-r*t)
    return K / denominator


def nxplot(G):
    """Plot graph G."""
    nx.draw_networkx(G)
    plt.show(block=False)

def nxdraw(g, offset=(0.01, -0.01)):
    # Obtain a layout for the graph.
    pos = nx.spring_layout(g)

    # Calculate off sets for the labels.
    x_shift, y_shift = offset
    posprime = {v: (x + x_shift, y + y_shift) for v, (x, y) in pos.items()}

    # Draw the graph and the labels.
    nx.draw_networkx( g, pos=pos, with_labels=False
                    , node_size=42
                    , node_color='#FFFFFF'
                    , edgecolors='#000000'
                    )
    nx.draw_networkx_labels( g, pos=posprime
                           , horizontalalignment='left'
                           , verticalalignment='top'
                           , font_size=8
                           , font_family='serif'
                           )

    # Open a window with the graph.
    plt.show()


def nxdraw_boxed(g):
    # Obtain a layout for the graph.
    pos = nx.random_layout(g)

    # Calculate off sets for the labels.
    x_shift, y_shift = 0, 0
    posprime = {v: (x + x_shift, y + y_shift) for v, (x, y) in pos.items()}

    # Draw the graph and the labels.
    nx.draw_networkx( g, pos=pos, with_labels=False
                    , node_size=42
                    , node_color='#FFFFFF'
                    , edgecolors='#000000'
                    , arrowsize=20
                    )
    nx.draw_networkx_labels( g, pos=posprime
                           , horizontalalignment='left'
                           , verticalalignment='top'
                           , font_size=8
                           , font_family='serif'
                           , bbox=dict(boxstyle="round", facecolor="white")
                           )

    # Open a window with the graph.
    plt.show()

def nxdraw_sub(g, v, depth=1):
    """Draw the `depth`-deep induced subgraph starting from vertex `v`."""
    # Make a list of the vertices involved.
    vertices = {v} # Don't forget the starting vertex.
    for n in range(depth):
        for v in vertices:
            newvs = {v for v in nx.all_neighbors(g, v)}
            vertices = vertices.union(newvs)
    nxdraw_boxed(g.subgraph(vertices))

nxopts = { 'node_size': 42
         , 'horizontalalignment': 'left'
         , 'verticalalignment': 'top'
         , 'font_size': '8'
         , 'font_family': 'serif'
         , 'node_color': '#FFFFFF'
         , 'edgecolors': '#000000'
         }

def nudge(pos, x_shift, y_shift):
    """...

    From https://stackoverflow.com/questions/14547388/networkx-in-python-draw-node-attributes-as-labels-outside-the-node by user scign.
    """
    return {n: (x+x_shift, y+y_shift) for n, (x, y) in pos.items()}

