from unravel import *

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import stopwords

def histplot(data):
    plt.hist(data, bins=data.max())
    plt.show()

# HILDA data for ISCO33: 106 rows, all 4203 columns.
h33 = hilda_by_isco(33)
# Subtract mean, divide by standard deviation.
h33scaled = StandardScaler().fit_transform(h33)
hscaled = StandardScaler().fit_transform(hilda)
# At this clustering depth: 1830 clusters, biggest is #3 with 705 features.
clustering = FeatureAgglomeration( n_clusters=None
                                 , distance_threshold=2).fit(h33)
# This seems better. Maximum at #20 with 1819 features but better distribution.
clustering = FeatureAgglomeration(n_clusters=100).fit(h33scaled)
clustering = FeatureAgglomeration(n_clusters=100).fit(hscaled)

clustered = clustering.fit_transform(hscaled)
data = pd.DataFrame( clustered
                   , columns=['C'+str(i) for i in range(clustered.shape[1])] )
# Cluster histogram.
np.unique(clustering.labels_, return_counts=True)

# def cols_in_cluster(clusterindex):
    # return [ meta.column_names_to_labels[col]
             # for col in h33.columns[np.where( clustering.labels_ == clusterindex
                                            # , True
                                            # , False )] ]
# def vars_in_cluster(index):
    # return [ col for col in h33.columns[np.where( clustering.labels_ == index
                                                # , True
                                                # , False )] ]

def cols_in_cluster( data # Data set (pandas dataframe)
                   , clustering # Clustering object.
                   , index # Index of cluster to find columns of.
                   ):
    return [ meta.column_names_to_labels[col]
             for col in data.columns[np.where( clustering.labels_ == index
                                             , True
                                             , False )] ]
def vars_in_cluster( data # Data set (pandas dataframe)
                   , clustering # Clustering object.
                   , index # Index of cluster to find columns of.
                   ):
    return [ col for col in data.columns[np.where( clustering.labels_ == index
                                                 , True
                                                 , False )] ]

def text_in_cluster(data, clustering, index):
    words = ' '.join(cols_in_cluster(data, clustering, index)).split()
    return [ word.lower() for word in words if word.isalpha() ]

def keywords(text, n=10, returndict=False):
    excludedwords = stopwords.get_stopwords('english')
    excludedwords.append('-')
    uwords = sorted( { word.lower()
                       for word in text
                       if word.lower() not in excludedwords } )
    d = dict( sorted( [ (word, text.count(word)) for word in uwords ]
                    , key=lambda t: t[1]
                    , reverse=True ) )
    if returndict: return d
    else: return list(d.keys())[:n]

def clusterkeywords(index, n=10, returndict=False):
    return keywords(text_in_cluster(index), n=n, returndict=returndict)


if __name__ == "__main__":
    # Job satisfaction and its effects.
    'ujbmsall' in vars_in_cluster(27)
    g = discover(['PC', 'GES', 'CCDr'], data.sample(2000))
    [ vertex for vertex in g.neighbors('C27') ] # -> C35
    [ vertex for vertex in g.neighbors('C35') ] # -> C93
    vertices = []
    vertices += vars_in_cluster(27)
    vertices += vars_in_cluster(35)
    vertices += vars_in_cluster(93)

    # List of list of variables.
    vertexlistoflists = [ vars_in_cluster(int(cluster[1:]))
                          for cluster in subgraph(g, 'C27', depth=4).nodes() ]
    vertices = [ vertex for sublist in vertexlistoflists for vertex in sublist ]


    h = discover(['PC', 'GES', 'CCDr'], hilda[vertices].sample(1000))
    [ meta.column_names_to_labels[var]
      for var in markov_blanket(h, 'ujbmsall').nodes() ]
