#!/bin/env python3
#
# unravel.py - causal modelling tools loader script.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2023-03-8
# Last modified: 2023-03-08
#

from unravel import *
from unravel.hilda import hilda, meta, fcols, cols, hildaf, hildab, hildaj


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
            run_algo_on_hilda(args[0])
