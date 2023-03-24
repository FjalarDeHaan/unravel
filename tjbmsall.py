# 2023-03-24

from unravel import *
from unravel.hilda import hilda, meta, fcols, cols, hildaf, hildab, hildaj, hilda1k, hilda100, hilda25, h100x300, h100x300_2

# Additive Noise Model for pairwise screening of potential causes and effects.
from cdt.causality.pairwise import ANM
anm = ANM()
# Build a dictionary of vars with ANM score of > .1 in abs value using hilda100.
d = {}
for i in range(hilda100.shape[1]):
    # Obtain ANM scores.
    score = anm.predict_proba((hilda100['tjbmsall'], hilda100.iloc[:, i]))
    # Weed out strangely high values (score ought to be between -1 and 1).
    # Keep only variables | scores | > .01 as potential causes or effects.
    if abs(score) < 1.5 and abs(score) > 0.01:
        d[hilda100.columns[i]] = score

# Reduce the candidate list by considering | score | > .1 variables only.
dsmaller = {key: val for (key, val) in d.items() if abs(val)> .1}
# Not to forget the focus of the analysis.
vertices = list(dsmaller) + ['tjbmsall']
# Run causal discovery algorithm on subset of candidate variables.
g = algos['GIES'].predict(hilda100[vertices])
# Make a dictionary of labels for the nodes.
dg = { col: meta.column_names_to_labels[col] for col in g.nodes() }
# Open a plot of the network of causes in a browser.
gplint(nx.relabel_nodes(causes(g, 'tjbmsall'), dg))

# d = { hilda100.columns[i]: anm.predict_proba((hilda100['tjbmsall'], hilda100.iloc[:, i]))
     # for i in range(hilda100.shape[1])
     # if abs(anm.predict_proba((hilda100['tjbmsall'], hilda100.iloc[:, i]))) < 1.5
     # if abs(anm.predict_proba((hilda100['tjbmsall'], hilda100.iloc[:, i]))) > .01 }
# h = algos['GIES'].predict(hilda100[list(dsmaller) + ['tjbmsall']])
# dh = { col: meta.column_names_to_labels[col] for col in h.nodes() }
# gplint(nx.relabel_nodes(causes(h, 'tjbmsall'), dh))

