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

def print_all_causal_paths( graph # Causal network.
                          , concepts # Dictionary of concepts => [ variables ].
                          , labels # Dictionary of variable => label
                          , target='all' # Effect concept to consider.
                          , include_empty=False # Whether to show empty paths.
                          ):
    if target == 'all':
        pairs = [ (from_concept, to_concept)
                  for from_concept in concepts
                  for to_concept in concepts
                  if from_concept != to_concept ]
    else:
        pairs = [ (from_concept, target)
                  for from_concept in concepts
                  for to_concept in [target]
                  if from_concept != to_concept ]
    for (from_concept, to_concept) in pairs:
        ps = causal_paths( graph
                         , concepts[from_concept]
                         , concepts[to_concept] )
        if include_empty or ps:
            print("#======================================#")
            print("# ", from_concept , "->" , to_concept)
            print("#======================================#")
        print_causal_paths( graph
                          , concepts[from_concept]
                          , concepts[to_concept]
                          , labels
                          , include_empty )

def print_causal_paths( graph
                      , from_vertices
                      , to_vertices
                      , labels
                      , include_empty=True ):
    # If a vertex set is just a single vertex, put it in a list anyway.
    if type(from_vertices) != list:
        from_vertices = [from_vertices]
    if type(to_vertices) != list:
        to_vertices = [to_vertices]
    ps = causal_paths(graph, from_vertices, to_vertices)
    if ps: # If there is any causal path at all...
        strength = sum([ 1 / (len(p)-1) for p in ps ])
        i = 0 # Path counter.
        for p in ps: # For each path...
            i += 1
            print( "[ %i / %i ]" % (i, len(ps))
                 , ", total causal strength = %f" % strength )
            for v in range(len(p)): # Treat each vertex in the path...
                if v == 0:
                    print("(*) ", end='')
                else:
                    print("==> ", end='')
                print(labels[p[v]])
            print()
    elif include_empty: # If there is no causal path and it needs to be shown.
        print( "[ 0 / 0 ], total causal strength = 0" )
        print()
    return len(ps)

def causal_paths(graph, from_vertices, to_vertices):
    paths = []
    for (fv, tv) in [ (fv, tv) for fv in from_vertices
                               for tv in to_vertices
                               if fv != tv ]:
        try:
            # TODO: Which is better?
            newpaths = list(nx.all_simple_paths(graph, fv, tv))
            # newpaths = list(nx.all_shortest_paths(graph, fv, tv))
        except nx.NetworkXNoPath:
            pass
        else:
            paths += newpaths
    return paths

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

def partrand(cs, n, var=None):
    """Return random partition of `cs` as list of lists of size `n`."""
    cs = list(cs)
    random.shuffle(cs)
    partition = [ cs[i:i+n] for i in range(0, len(cs), n) ]
    # If the last list is too small, merge it with the one before.
    if len(partition[-1]) < n // 2:
        merged = partition[-2] + partition[-1]
        partition = partition[:-2]
        partition.append(merged)
    # If only a partition is wanted, all is done.
    if var is None:
        return partition
    # Otherwise make sure all parts contain `var`.
    else:
        for part in partition:
            if var not in part:
                part.append(var)
        return partition

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

def markov_blankets(graph, vertices):
    subgraphs = [ markov_blanket(graph, vertex) for vertex in vertices ]
    return nx.compose_all(subgraphs)

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
    children = list(digraph.successors(vertex))
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
