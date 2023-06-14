#
# benchmark.py - tools for benchmarking causal discovery algorithms.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-06-02
# Last modified: 2023-06-02
#
from unravel.causal import *


import time
import multiprocessing

import cdt
from cdt.data import AcyclicGraphGenerator as DAG

import numpy as np
import pandas as pd


# Synthetic data parameters.
nvertices = [ 10, 50, 100, 500 ]
nrows = [ 100, 500, 1000 ]

def errors(bs): return [ not b for b in bs ]
def errorcount(bs): return sum(errors(bs))

def VHD(trial, truth, average=True):
    """Compute vertex-based hamming distances between `trial` and `truth`."""
    # Prepare a vertex->score dictionary.
    ds = {}
    # Check each vertex.
    for vertex in truth.nodes():
        score = 0
        # How many in-edges missed?
        score += errorcount([ trial.has_edge(*edge)
                              for edge in truth.in_edges(vertex) ])
        # How many in-edges made up?
        score += errorcount([ truth.has_edge(*edge)
                              for edge in trial.in_edges(vertex) ])
        # How many out-edges missed?
        score += errorcount([ trial.has_edge(*edge)
                              for edge in truth.out_edges(vertex) ])
        # How many out-edges made up?
        score += errorcount([ truth.has_edge(*edge)
                              for edge in trial.out_edges(vertex) ])
        # Provide the score as well as the true number of edges.
        ds[vertex] = score, truth.degree(vertex)
    # Deliver.
    if average:
        values = [ t[0] for t in ds.values() ]
        return sum(values) / len(ds)
    else:
        return ds

def precision(trial, truth, average=True):
    """Return fraction of true edges and all edges in `trial`, by vertex."""
    # Prepare a vertex->score dictionary.
    ds = {}
    # Check each vertex.
    for vertex in truth.nodes():
        score = 0
        # How many true in-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.in_edges(vertex) ])
        # How many true out-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.out_edges(vertex) ])
        # Compute the precision.
        # TODO: Check the below --- trial.degree?!
        # if score == 0 and trial.degree(vertex) == 0:
            # ds[vertex] = 1.0
        if score > 0:
            ds[vertex] = score / trial.degree(vertex)
    # Deliver.
    if average:
        return sum(ds.values()) / len(ds)
    else:
        return ds

def recall(trial, truth, average=True):
    """Return fraction of all true edges found in `trial`, by vertex."""
    # Prepare a vertex->score dictionary.
    ds = {}
    # Check each vertex.
    for vertex in truth.nodes():
        score = 0
        # How many true in-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.in_edges(vertex) ])
        # How many true out-edges?
        score += sum([ trial.has_edge(*edge)
                       for edge in truth.out_edges(vertex) ])
        # Compute the recall.
        if score == 0 and truth.degree(vertex) == 0:
            ds[vertex] = 1.0
        else:
            ds[vertex] = score / truth.degree(vertex)
    # Deliver.
    if average:
        return sum(ds.values()) / len(ds)
    else:
        return ds

def rel_edge_error(trial, truth):
    """Return average and std dev of edge error per true edge."""
    average = np.mean([ score[0]/score[1]
                        for (vertex, score) in VHD(g, graph).items()
                        if score[1] != 0 ] # Don't count isolated vertices.
                     )
    stddev = np.std([ score[0]/score[1]
                      for (vertex, score) in VHD(g, graph).items()
                      if score[1] != 0 ] # Don't count isolated vertices.
                   )
    return average, stddev

def benchmark( algo # Pass either a string or an actual, instantiated algo.
             , nvertices
             , nrows
             , iterations=11
             , mechanism='linear'
             , noise='gaussian' ):
    """Compute a suite of benchmarks."""
    # Prepare a metric->score dictionary.
    benchmarks = {}
    # Initialise variables.
    prc =  rec = vhd = shd = sid = 0.0
    # Iterate to get credible statistics.
    for i in range(iterations):
        data, truth = generate( mechanism
                              , noise=noise
                              , nvertices=nvertices
                              , nrows=nrows)
        print("Discovering causal graph for iteration %i." % i)
        if type(algo) == str:
            trial = algos[algo].predict(data)
        else:
            trial = algo.predict(data)
        # Calculate the statistics.
        prc += precision(trial, truth)
        rec += recall(trial, truth)
        vhd += VHD(trial, truth)
        shd += SHD(trial, truth)
        sid += SID(trial, truth)
    # Do the averaging.
    benchmarks["precision"] = prc / iterations
    benchmarks["recall"] = rec / iterations
    benchmarks["VHD"] = vhd / iterations
    benchmarks["SHD"] = shd / iterations
    benchmarks["SID"] = sid / iterations
    # Deliver.
    return benchmarks

def bmarkinsect( algolist
               , nvertices
               , nrows
               , iterations=11
               , mechanism='linear'
               , noise='gaussian'
               , chunksize=None ):
    """Compute a suite of benchmarks."""
    # Prepare a metric->score dictionary.
    benchmarks = {}
    # Initialise variables.
    prc =  rec = vhd = shd = sid = 0.0
    # Prepare empty lists for graphs and datasets.
    truths = []
    datas = []
    trials = []
    # Iterate to get credible statistics.
    for i in range(iterations):
        data, truth = generate( mechanism
                              , noise=noise
                              , nvertices=nvertices
                              , nrows=nrows)
        print("Discovering causal graphs for iteration %i." % i)
        trial = discover(algolist, data, chunksize)
        # Archive the graphs and data.
        truths.append(truth)
        trials.append(trial)
        datas.append(data)
        # Calculate the statistics.
        prc += precision(trial, truth)
        rec += recall(trial, truth)
        vhd += VHD(trial, truth)
        shd += SHD(trial, truth)
        sid += SID(trial, truth)
    # Do the averaging.
    benchmarks["precision"] = prc / iterations
    benchmarks["recall"] = rec / iterations
    benchmarks["VHD"] = vhd / iterations
    benchmarks["SHD"] = shd / iterations
    benchmarks["SID"] = sid / iterations
    benchmarks["truths"] = truths
    benchmarks["trials"] = trials
    benchmarks["datas"] = datas
    # Deliver.
    return benchmarks

def discover(algolist, data, chunksize=None):
    # In case just one algo is passed, put it in a list anyway.
    if type(algolist) == str: algolist = [algolist]
    # Prepare an empty list to hold the discovered causal graphs' edges.
    edgesets = []
    # Add causal graphs returned by each algorithm.
    for algo in algolist:
        # Only do it chunkedly if asked and necessary.
        if chunksize is None or chunksize >= data.shape[0]:
            print("Running %s algorithm." % algo)
            graph = algos[algo].predict(data)
        else:
            print("Running %s algorithm in chunks of %i." % ( algo
                                                            , chunksize))
            chunks = partrand(data.columns, chunksize)
            graphs = [ algos[algo].predict(data[chunk]) for chunk in chunks ]
            graph = nx.compose_all(graphs)
        # Add the edgeset of the discovered graph to the list.
        edgesets.append(graph.edges)
    # Collect the intersection of the edge-sets.
    sharededges = set.intersection(*map(set, edgesets))
    # Construct the graph from the intersection.
    graph = nx.DiGraph()
    graph.add_nodes_from(data.columns)
    graph.add_edges_from(sharededges)
    # Deliver.
    return graph

def generate( mechanism='linear'
            , noise='gaussian'
            , nvertices=nvertices
            , nrows=nrows ):
    """Generate synthetic data and corresponding 'ground-truth' causal graph."""
    generator = DAG(mechanism, noise=noise, nodes=nvertices, npoints=nrows)
    data, truth = generator.generate()
    return data, truth


def tonetworkx(clgraph):
    """Convert a Causal-Learn graph object to a NetworkX graph object."""
    # Populate the internal NetworkX DiGraph object.
    clgraph.to_nx_graph()
    # Get it out of the Causal-Learn graph object.
    g = clgraph.nx_graph
    # Create a old label to new label dictionary.
    d = { vertex: 'V' + str(vertex) for vertex in gprime.nodes() }
    # Relabel and overwrite graph.
    g = nx.relabel_nodes(g, d)
    # Deliver.
    return g









