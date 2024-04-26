#!/bin/env python3
#
# unravel.py - causal modelling tools loader script.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-03-8
# Last modified: 2023-04-25
#

from unravel import *
from unravel.hilda import *

import multiprocessing
import copy


def cli_args():
    """Parse `argv` interpreter-agnostically. Return non-trivial arguments."""
    argv = os.sys.argv
    fname = os.path.basename(__file__)
    for arg in argv:
        if arg.endswith(fname):
            index = argv.index(arg) + 1
    if len(argv) > index:
        return argv[index:]
    else:
        return []

def run_algo_on_hilda(algo):
    """Run algorithm `algo` on HILDA and write causal graph to pickle."""
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda)
    pname = "graph-" + algo + ".pickle"
    with open(pname, "wb") as f:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f)

def run_algo_on_hilda1k(algo):
    """Run algorithm `algo` on HILDA and write causal graph to pickle."""
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda1k)
    pname = "graph-" + algo + "-hilda1k" + ".pickle"
    with open(pname, "wb") as f:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f)

def run_algo_on_hilda100(algo):
    """Run algorithm `algo` on HILDA and write causal graph to pickle."""
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda1k)
    pname = "graph-" + algo + "-hilda100" + ".pickle"
    with open(pname, "wb") as f:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f)

def run_algo_on_candidates(algo):
    with open("../analysis-hilda/candidates-20230406.pickle", "rb") as f1:
        cs = pickle.load(f1)
    candidates = cs + bcols
    print("Running %s algorithm." % algo)
    g = algos[algo].predict(hilda1k[candidates])
    pname = "graph-" + algo + "-hilda100" + ".pickle"
    with open(pname, "wb") as f2:
        print("Writing output of %s to %s." % (algo, pname))
        pickle.dump(g, f2)

def load_analysis():
    global g, h, dg, dh, dcauses, deffects, catcauses, cateffects
    with open( "/home/fjalar/pCloudDrive/archive/academia/projects/"
               "future-of-work/analysis-hilda/"
               "graph-GIES-hilda100-bcols.pickle", "rb") as f:
        g = pickle.load(f)
    with open( "/home/fjalar/pCloudDrive/archive/academia/projects/"
               "future-of-work/analysis-hilda/"
               "graph-GIES-hilda100-tjbmsall.pickle", "rb") as f:
        h = pickle.load(f)
    dg = { col: meta.column_names_to_labels[col] for col in g.nodes() }
    dh = { col: meta.column_names_to_labels[col] for col in h.nodes() }
    gplint(nx.relabel_nodes(causes(g, 'tjbmsall'), dg))
    gplint(nx.relabel_nodes(causes(h, 'tjbmsall'), dh))
    dcauses= nx.shortest_path_length(g, target='tjbmsall')
    deffects = nx.shortest_path_length(g, source='tjbmsall')
    cateffects = { cat: np.mean( [ deffects[v] for v in contractions[cat]
                                                if v in deffects ] )
                  for cat in contractions }
    catcauses = { cat: np.mean( [ dcauses[v] for v in contractions[cat]
                                              if v in dcauses ] )
                  for cat in contractions }

def run_isco(algo='GIES', isco=None):
    if isco is None:
        print("Error: ISCO code not provided.")
    else:
        print("Subsetting HILDA to ISCO code %i." % isco)
        h = hilda_by_isco(isco)
        print("Getting candidate causes and effects.")
        cs = candidates(bcols, h)
        print("Discovering causal graph.")
        g = algos[algo].predict(h[cs + bcols])
        # Write the causal graph to a pickle.
        pname = "graph-isco" + str(isco) + ".pickle"
        with open(pname, "wb") as f:
            print("Writing causal graph to %s." % pname)
            pickle.dump(g, f)
        return g

def run_stratified(algo='GIES', iscos=iscover100):
    # Prepare a dictionary for the causal graphs.
    d = {}
    # It is okay to pass a single code not wrapped in a list.
    if type(iscos) != list: iscos = [iscos]
    # Put causal graph in dictionary keyed by ISCO code.
    for isco in iscos:
        d[isco] = run_isco(algo=algo, isco=isco)
    # Deliver.
    with open("graphs-by-isco-dict.pickle", "wb") as f: pickle.dump(d, f)
    return d

def run_stratified_parallel(algo='GIES', iscos=iscover100):
    # Set up the multiprocessing facilities.
    pool = multiprocessing.Pool()
    # Collect the multiprocessing pre-results.
    d = { isco: pool.apply_async(run_isco, (algo, isco)) for isco in iscos }
    # Then extract the actual causal graphs and return them.
    return { isco: d[isco].get() for isco in iscos }

def run_isco_colsampled(algo='GIES', isco=None, ncols=10, niters=10):
    if isco is None:
        print("Error: ISCO code not provided.")
    else:
        print("Subsetting HILDA to ISCO code %i." % isco)
        h = hilda_by_isco(isco)
        gs = []
        i = 0
        while len(gs) < niters:
            hsample = h.sample(n=ncols, axis=1, random_state=i)
            if 'ujbmsall' in hsample:
                print("Discovering causal graph number %i." % len(gs))
                g = algos[algo].predict(hsample)
                gs.append(g)
                print("Appended graph.")
            i += 1
        composition = gs[0]
        for g in gs[1:]:
            composition = nx.compose(composition, g)

        return gs, composition

def collapsed_graph( data # Sub-set of hilda data.
                   , algos = ['PC', 'GES'] # List of algorithms to use.
                   , probability = .6 # Likelihood of existence of edge.
                   , iterations = 100 # How many times to sample in MC process.
                   ):
    # Get the intersection of the causal graphs from the list algorithms.
    g = discover(algos, data[c2h.variable.to_list()], intersected=True)
    # Compute edge list with MC probabilities for collapsed graph.
    A = [ (c1, c2, mcprob( g
                         , concepts[c1], concepts[c2]
                         , probability=.6, iterations=100 ))
          for c1 in concepts for c2 in concepts if c1 != c2 ]
    # Prepare an empty directed graph.
    dg = nx.DiGraph()
    # Add the edges from the edge list.
    dg.add_weighted_edges_from(A)
    # Deliver.
    return dg

combinations = { 'sex': [ "any", "male", "female" ]
               , 'age': [ "any"
                        , "0-9"
                        , "10-19"
                        , "20-29"
                        , "30-39"
                        , "40-49"
                        , "50-59"
                        , "60-69"
                        , "70-79"
                        , "80-89"
                        , "90-99"
                        , "100-110" ]
                , 'isco': ["any"] + iscover100
                , 'education': ["any"] + [i for i in range(1, 11)]
                , 'seifa': ["any"] +[i for i in range(1, 11)]
                }

def subsethilda( sex = "any"
               , age = "any"
               , isco = "any"
               , education = "any"
               , seifa = "any" ):
    # ISCO first and making sure none of this is destructive.
    if isco == "any":
        h = copy.deepcopy(hilda)
    elif isco in iscover100:
        h = copy.deepcopy(hilda_by_isco(isco))
    else:
        print("Error: wrong isco key.")
    # Filter on 'sex'.
    if sex == "any":
        pass
    elif sex == 'male':
        h = h[h['uhgsex'] == 1]
    elif sex == 'female':
        h = h[h['uhgsex'] == 2]
    else:
        print("Error: wrong sex key.")
    # Filter on 'age'.
    if age == "any":
        pass
    elif age == "0-9":
        h = h[h[concepts['age'][0]].between(0, 10, inclusive='left')]
    elif age == "10-19":
        h = h[h[concepts['age'][0]].between(10, 20, inclusive='left')]
    elif age == "20-29":
        h = h[h[concepts['age'][0]].between(20, 30, inclusive='left')]
    elif age == "30-39":
        h = h[h[concepts['age'][0]].between(30, 40, inclusive='left')]
    elif age == "40-49":
        h = h[h[concepts['age'][0]].between(40, 50, inclusive='left')]
    elif age == "50-59":
        h = h[h[concepts['age'][0]].between(50, 60, inclusive='left')]
    elif age == "60-69":
        h = h[h[concepts['age'][0]].between(60, 70, inclusive='left')]
    elif age == "70-79":
        h = h[h[concepts['age'][0]].between(70, 80, inclusive='left')]
    elif age == "80-89":
        h = h[h[concepts['age'][0]].between(80, 90, inclusive='left')]
    elif age == "90-99":
        h = h[h[concepts['age'][0]].between(90, 100, inclusive='left')]
    elif age == "100+":
        h = h[h[concepts['age'][0]] >= 100]
    else:
        print("Error: wrong age key.")
    # Filter on 'education'.
    if education == "any":
        pass
    elif education in [i for i in range(1, 11)]:
        h = h[ h[concepts['education'][0]] == education ]
    else:
        print("Error: wrong education key.")
    # Filter on 'seifa'.
    if seifa == "any":
        pass
    elif seifa in [i for i in range(1, 11)]:
        h = h[ h[concepts['seifa'][0]] == seifa ]
    else:
        print("Error: wrong seifa key.")
    # Deliver.
    return h

graphkeys = [ (sex, age, isco, education, seifa)
              for sex in combinations['sex']
              for age in combinations['age']
              for isco in combinations['isco']
              for education in combinations['education']
              for seifa in combinations['seifa'] ]

def graphdict():
    return { gk: collapsed_graph(subsethilda(*gk)) for gk in graphkeys }


if __name__ == '__main__':
    args = cli_args()
    if len(args) > 0:
        print(args)
        if args[0] == 'blankets':
            # memory_limit()
            bs = blankets(hilda, list(cols.keys()), parallel=False)
            # bs = blankets(hilda, list(cols.keys()), parallel=True)
            with open("blankets.pickle", "wb") as f: pickle.dump(bs, f)
        else:
            run_stratified_parallel(args[0])
    else:
        pass
