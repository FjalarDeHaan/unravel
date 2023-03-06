#!/bin/env python3
#
# gplot.py - some utilities for plotting graphs.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-02-14
# Last modified: 2023-02-14
#

import networkx as nx

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pyvis
from pyvis.network import Network

def gplot(g, offset=(0.01, -0.01), boxed=True, layout='random'):
    # Obtain a layout for the graph.
    if   layout == 'circular': pos = nx.circular_layout(g)
    elif layout == 'kk': pos = nx.kamada_kawai_layout(g)
    elif layout == 'shell': pos = nx.shell_layout(g)
    elif layout == 'spring': pos = nx.spring_layout(g)
    elif layout == 'spectral': pos = nx.spectral_layout(g)
    elif layout == 'spiral': pos = nx.spiral_layout(g)
    else: pos = nx.random_layout(g)

    # Calculate off sets for the labels.
    if boxed:
        x_shift, y_shift = 0, 0
    else:
        x_shift, y_shift = offset
    posprime = {v: (x + x_shift, y + y_shift) for v, (x, y) in pos.items()}

    # Draw the graph and the labels.
    if boxed:
        nx.draw_networkx( g, pos=pos, with_labels=False
                        , node_size=42
                        , node_color='#FFFFFF'
                        , linewidths=0
                        , edgecolors='#000000'
                        #, arrowstyle='fancy'
                        )
        nx.draw_networkx_labels( g, pos=posprime
                            , horizontalalignment='left'
                            , verticalalignment='top'
                            , font_size=8
                            , font_family='serif'
                            , bbox=dict(boxstyle="round", facecolor="white")
                            )
    else:
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

def subg(g, v, depth=1):
    """Return the `depth`-deep induced subgraph starting from vertex `v`."""
    # Make a list of the vertices involved.
    vertices = {v} # Don't forget the starting vertex.
    for n in range(depth):
        for v in vertices:
            newvs = {v for v in nx.all_neighbors(g, v)}
            vertices = vertices.union(newvs)
    return g.subgraph(vertices)

def gplint(g, fname='graph.html'):
    # Goth mode for display on monitors rather than paper.
    net = Network( directed=True
                 , bgcolor='#000000'
                 , font_color='#CCCCCC'
                 , height='900px'
                 , width='1600px'
                 , select_menu=True
                 , filter_menu=True
                 )
    net.from_nx(g)
    net.toggle_physics(False)
    net.show_buttons()
    net.show(fname)
