#!/bin/env python3
#
# gtools.py - some utilities for plotting graphs.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-02-14
# Last modified: 2023-03-08
#

import networkx as nx

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pyvis.network import Network


def clean_edge_props(graph):
    """Destructively remove all edge properties save `width`."""
    # Loop through all edges as (from, to, contraction) triples.
    for edge in graph.edges.data('contraction'):
        # If there actually is contraction information.
        if edge[2] is not None:
            del graph.edges[edge[0], edge[1]]['contraction']

def contract(graph, vertices, label=None):
    """Contract `vertices` into one labelled `vertices[0] or `label`."""
    # Make very sure this function is non-destructive.
    graph = copy.deepcopy(graph)
    # Prepare the label for the contracted vertex.
    if label is None:
        # Use first vertex label for the contracted vertex.
        label = vertices[0]
    else:
        # Rename first vertex in graph to use `label` for contracted vertex.
        nx.relabel_nodes(graph, {vertices[0]: label}, copy=False)
    # Contract the vertices with the first, one by one consuming the list.
    for vertex in vertices[1:]:
        nx.contracted_nodes(graph, label, vertex, self_loops=False, copy=False)
    # Get rid of 'contraction' edge labels so `gplint()` does not get confused.
    clean_edge_props(graph)
    return graph

def markov_blanket(graph, vertex):
    if nx.is_directed(graph):
        parents = list(graph.predecessors(vertex))
        children = list(graph.successors(vertex))
        spouses = []
        for child in children:
            spouses += list(graph.predecessors(child))
        blanket = parents + children + spouses + [vertex]
    else:
        blanket = []
        neighbours = list(graph.neighbors(vertex))
        for neighbour in neighbours:
            blanket += list(graph.neighbors(neighbour))
    return graph.subgraph(blanket)

def ingraph(digraph, vertex):
    """Return subgraph induced by `vertex` and vertices adjacent _to_ it."""
    vertices = {vertex}
    vertices = vertices.union(digraph.predecessors(vertex))
    return digraph.subgraph(vertices)

def outgraph(digraph, vertex):
    """Return subgraph induced by `vertex` and vertices adjacent _from_ it."""
    vertices = {vertex}
    vertices = vertices.union(digraph.successors(vertex))
    return digraph.subgraph(vertices)

def causes(digraph, vertex):
    """Return subgraph induced by causes of `vertex` (identical `ingraph()`)."""
    return ingraph(digraph, vertex)

def effects(digraph, vertex):
    """Return subgraph induced by effects of `vertex` (out-Markov blanket)."""
    children = digraph.successors(vertex)
    spouses = []
    for child in children:
        spouses += digraph.predecessors(child)
    effects = set(children).union(set(spouses))
    return digraph.subgraph(effects)

def subgraph(g, v, depth=1):
    """Return the `depth`-deep induced subgraph starting from vertex `v`."""
    # Make a list of the vertices involved.
    vertices = {v} # Don't forget the starting vertex.
    for n in range(depth):
        for v in vertices:
            newvs = {v for v in nx.all_neighbors(g, v)}
            vertices = vertices.union(newvs)
    return g.subgraph(vertices)

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
