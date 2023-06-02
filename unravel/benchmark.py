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
        if score == 0 and trial.degree(vertex) == 0:
            ds[vertex] = 1.0
        else:
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

def benchmark( algo
             , nvertices
             , nrows
             , iterations=11
             , mechanism='linear'
             , noise='gaussian' ):
    """Compute a suite of benchmarks."""
    # Prepare a metric->score dictionary.
    benchmarks = {}
    # Iterate to get credible statistics.
    prc = 0.0
    rec = 0.0
    vhd = 0.0
    for i in range(iterations):
        generator = DAG(mechanism, noise=noise, nodes=nvertices, npoints=nrows)
        data, truth = generator.generate()
        print("Discovering causal graph for iteration %i." % i)
        trial = algos[algo].predict(data)
        prc += precision(trial, truth)
        rec += recall(trial, truth)
        vhd += VHD(trial, truth)
    benchmarks["precision"] = prc / iterations
    benchmarks["recall"] = rec / iterations
    benchmarks["VHD"] = vhd / iterations
    benchmarks["SHD"] = SHD(trial, truth)
    benchmarks["SID"] = SID(trial, truth)
    return benchmarks










