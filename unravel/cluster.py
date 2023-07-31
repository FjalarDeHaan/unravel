from unravel import *

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import stopwords
import difflib
import Levenshtein

def histplot(data):
    plt.hist(data, bins=data.max())
    plt.show()

def d(s1, s2):
    return 1 - difflib.SequenceMatcher( None
                                      , s1.lower().split()
                                      , s2.lower().split() ).ratio()

def cull(data):
    cols = data.columns.to_list()
    # Remove relationship grid columns: 57
    cs1 = {col for col in cols if 'urx' in col}
    # 'Replicate weight' columns: 270
    cs2 = { col for col in cols
                if 'Replicate weight' in meta.column_names_to_labels[col] }
    # 'Imputation flag' columns: 125
    cs3 = { col for col in cols
                if 'Imputation flag' in meta.column_names_to_labels[col] }
    # 'Population weight' columns: 8
    cs4 = { col for col in cols
                if 'Population weight' in meta.column_names_to_labels[col] }
    # 'interview outcome' columns: 30
    cs5 = { col for col in cols
                if 'interview outcome' in meta.column_names_to_labels[col] }
    # '(relationship)' columns: 45
    cs6 = { col for col in cols
                if '(relationship)' in meta.column_names_to_labels[col] }
    # 'Relationship in household' columns: 10
    cs7 = { col for col in cols
                if 'Relationship in household'
                in meta.column_names_to_labels[col] }
    # 'Income unit' columns: 11
    cs8 = { col for col in cols
                if 'Income unit' in meta.column_names_to_labels[col] }
    # 'Family type' columns: 10
    cs9 = { col for col in cols
                if 'Family type' in meta.column_names_to_labels[col] }
    # 'Family number person' columns: 11
    cs10 = { col for col in cols
                if 'Family number person' in meta.column_names_to_labels[col] }
    # 'Relationship of self' columns: 13
    cs11 = { col for col in cols
                 if 'Relationship of self' in meta.column_names_to_labels[col] }
    # 'Enumerated person' columns: 4
    cs12 = { col for col in cols
                 if 'Enumerated person' in meta.column_names_to_labels[col] }
    # 'Imputed age' columns: 8
    cs13 = { col for col in cols
                 if 'Imputed age' in meta.column_names_to_labels[col] }
    # 'Wave last interviewed' columns: 8
    cs14 = { col for col in cols
                 if 'Wave last interviewed' in meta.column_names_to_labels[col] }
    tocull = set.union( cs1, cs2, cs3, cs4, cs5
                      , cs6, cs7, cs8, cs9, cs10
                      , cs11, cs12, cs13, cs14
                      )
    for col in tocull: cols.remove(col)
    return data[cols], len(tocull)

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

def pad(s1, s2):
    """Pad shortest string with spaces, return adjusted pair."""
    # Equal length strings. Do nothing.
    if len(s1) == len(s2): return s1, s2
    # String `s1` shorter. Pad it.
    elif len(s1) < len(s2):
        d = len(s2) - len(s1)
        return s1 + d*' ', s2
    # String `s2` shorter. Pad it.
    elif len(s1) > len(s2):
        d = len(s1) - len(s2)
        return s1, s2 + d*' '

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

def dmatrix(labels):
    A = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        for j in range(1+i, len(labels)):
            A[i, j] = A[j, i] = d(labels[i], labels[j])
    return A

if __name__ == "__main__":
    h, _ = cull(hilda)
    h33, _ = cull(hilda_by_isco(33))
    labels = np.array( [ meta.column_names_to_labels[col]
                         for col in h.columns ] )
    A = dmatrix(labels)
    clustering = OPTICS(metric='precomputed').fit(A)

    clustersize = len(labels)
    iteration = 1
    while clustersize > 399:
        # Find largest cluster.
        bins, contents = np.unique(clustering.labels_, return_counts=True)
        argmx = contents.argmax()
        clusterindex = bins[argmx]     # Index of biggest cluster.
        clustersize = contents[argmx]  # Size of biggest cluster.
        lowestindex = bins.min()       # Lowest bin index, e.g. -1.
        highestindex = bins.max()      # Highest bin index, e.g. 254.
        print( "Largest cluster is %i, containing %i variables."
             % (clusterindex, clustersize) )
        print( "Recursively clustering largest cluster. Iteration: %i"
             % iteration)
        iteration += 1
        B = dmatrix(labels[clustering.labels_ == clusterindex])
        clusteringB = OPTICS( min_samples=2
                            , metric='precomputed').fit(B)
        # If only one cluster is found, report and break loop.
        if np.unique(clusteringB.labels_).shape[0] == 1:
            print("Only one cluster found. Nothing to be done.")
            break
        else:
            # Outliers are different outliers than before, thus different bin.
            mask = np.where( clusteringB.labels_ == -1
                           , lowestindex - 1
                           , clusteringB.labels_ )
            # First bin (i.e. index zero) goes in previous bin, rest to end.
            mask = np.where( mask == 0
                           , clusterindex
                           , highestindex + mask )
            # Apply.
            clustering.labels_[clustering.labels_ == clusterindex] = mask









