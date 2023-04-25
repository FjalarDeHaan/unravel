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
        print("Subsetting HILDA to ISCO code %i" % isco)
        h = hilda_by_isco(isco)
        print("Getting candidate causes and effects.")
        cs = candidates(bcols, h)
        print("Discovering causal graph.")
        g = algos[algo].predict(cs + bcols)
        # Write the causal graph to a pickle.
        pname = "graph-isco" + isco + ".pickle"
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
            run_stratified(args[0])
    else:
        pass
        # load_analysis()

