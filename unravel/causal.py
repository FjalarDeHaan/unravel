#!/bin/env python3
#
# causal.py - causal modelling with HILDA data.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-02-14
# Last modified: 2023-03-08
#

import pickle, os, multiprocessing, copy

import cdt
from cdt.metrics import SHD, SID

from pyCausalFS.CBD.MBs.HITON.HITON_MB import HITON_MB

import networkx as nx
import pandas as pd
import numpy as np


glasso = cdt.independence.graph.Glasso()

# Make sure SID returns an integer.
def SID(target, prediction): return int(cdt.metrics.SID(target, prediction))

# Instantiate all graph-based algorithms.
algorithms = [ cdt.causality.graph.CAM()
             , cdt.causality.graph.CCDr()
             , cdt.causality.graph.GES()
             , cdt.causality.graph.GIES()
             , cdt.causality.graph.LiNGAM()
             , cdt.causality.graph.PC()
             , cdt.causality.graph.SAM()
             , cdt.causality.graph.SAMv1() ]
algos = { str(algo).split(sep='.')[3]: algo for algo in algorithms }
nalgos = len(algos)

def distances(data, algos):
    algonames = list(algos.keys())
    nalgos = len(algonames)
    df = pd.DataFrame(index=algonames, columns=algonames)
    gs = [ algos[algo].predict(data) for algo in algos ]
    shd_matrix = df.copy()
    sid_matrix = df.copy()
    for row in range(nalgos):
        for col in range(nalgos):
            shd_matrix.loc[algonames[row], algonames[col]]=SHD(gs[row], gs[col])
            sid_matrix.loc[algonames[row], algonames[col]]=SID(gs[row], gs[col])
    return shd_matrix, sid_matrix

def blanket(data, var, algorithm=HITON_MB, alpha=.01):
    """Extract Markov blanket incl. seed var. as label list."""
    # Extract column names.
    cols = data.columns
    # Get index of variable `var`.
    index = cols.to_list().index(var)
    # Obtain Markov blanket of `var`.
    b = algorithm(data, index, alpha)[0]
    # Replace indices with labels.
    variables = list(cols[b])
    # Add source variable and return blanket as list of variable labels.
    return [var] + variables

def blankets(data, variables, algorithm=HITON_MB, alpha=.01, parallel=False):
    """Extract Markov blankets incl. seeds of each variable."""
    if parallel:
        pool = multiprocessing.Pool()
        blankets = {}
        for variable in variables:
            args = data, variable, algorithm, alpha
            blankets[variable] = pool.apply_async(blanket, args)
        blankets = { key: val.get() for (key, val) in blankets.items() }
    else:
        blankets = { variable: blanket(data, variable, algorithm, alpha)
                     for variable in variables }
    return blankets

def causal_blanket(data, variables, algo='GES', alpha=.01):
    """Do causal discovery on Markov blanket of `var`."""
    # If `var` is a list, assume it is a list of seed variables.
    if type(variables) == list:
        b = []
        for var in variables:
            b += blanket(data, var, alpha=alpha)
    # If it ain't, assume it is the label of a single seed variable.
    else:
        # First` get the blanket, including the seed variable.
        b = blanket(data, variables, alpha=alpha)
    # Avoid repetition.
    b = set(b)
    # Then subset the data.
    subdata = data[b]
    # Run causal discovery algorithm on blanket data only and return result.
    return algos[algo].predict(subdata)
