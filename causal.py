#!/bin/env python3
#
# causal.py - causal modelling with HILDA data.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-02-14
# Last modified: 2023-02-14
#

import cdt
from cdt.metrics import SHD
from cdt.metrics import SID

import dowhy
import dowhy.gcm
from dowhy.gcm import EmpiricalDistribution, AdditiveNoiseModel
from dowhy.gcm.ml import create_linear_regressor

import networkx as nx

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, uniform, expon

from gplot import *
from hilda import hilda, fcols

# Use only the columns in `fcols` and get rid of the NaNs naively.
data = hilda[fcols.keys()].dropna()

glasso = cdt.independence.graph.Glasso()

# Make sure SID returns an integer.
def SID(target, prediction): return int(cdt.metrics.SID(target, prediction))

# Instantiate all graph-based algorithms.
cam    = cdt.causality.graph.CAM()
ccdr   = cdt.causality.graph.CCDr()
ges    = cdt.causality.graph.GES()
gies   = cdt.causality.graph.GIES()
lingam = cdt.causality.graph.LiNGAM()
pc     = cdt.causality.graph.PC()
sam    = cdt.causality.graph.SAM()
samv1  = cdt.causality.graph.SAMv1()
algos = [ # cam
         ccdr
        , ges
        , gies
        # , lingam
        , pc
        , sam
        , samv1 ]

algonames = [ str(algo).split(sep='.')[3] for algo in algos ]
df = pd.DataFrame(index=algonames, columns=algonames)
# gs = [ algo.predict(data) for algo in algos ]

