import pandas as pd
from nltk import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering


def clustering(data, n, m):
    """Clustering algorithm
    Input:
    data ():
    n (int): number of clusters
    m (str): representation method
    Output:
    Predicted labels of clusters, matrix of clusters
    """

    # first case: tf-idf method
    if m == 'tf-idf':
        # creating the vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=8000,
                                           use_idf=True,
                                           stop_words='english',
                                           tokenizer=word_tokenize,
                                           ngram_range=(1, 3))
        # creating the matrix
        matrix = tfidf_vectorizer.fit_transform(data).todense()

    # second case: token frequency
    elif m == 'token frequency':
        # creating the vectorizer
        vectorizer = CountVectorizer(binary=False)
        # ???
        x_count = vectorizer.fit_transform(data)
        # creating the matrix
        matrix = x_count.todense()

    # third case: tokens
    elif m == 'tokens':
        # creating the vectorizer
        vectorizer = CountVectorizer(binary=True)
        # ???
        x_count = vectorizer.fit_transform(data)
        # creating the matrix
        matrix = x_count.todense()

    # after having tried with KMeans, we observed better results for Agglomerative clustering
    #km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=30, verbose=0, random_state=3425)
    #km.fit(matrix)

    # using the agglomerative clustering algorithm
    ac = AgglomerativeClustering(n_clusters=n, affinity='euclidean', memory=None, connectivity=None,
                                 compute_full_tree='auto', linkage='ward', distance_threshold=None,
                                 compute_distances=False)
    # fitting the matrix
    ac.fit(matrix)
    # renaming the labels
    pred_labels = ac.labels_
    return pred_labels, matrix, n


def scores(pred_labels, matrix, n):
    """
    Compute evaluation scores:
    Output: silhouette coeff, homogeneity, completeness, v-measure, adjusted Rand index
    """

    # calling this score first, as it does not depend on the labels
    sil = metrics.silhouette_score(matrix, pred_labels, sample_size=1000)

    # if there are 6 clusters, the labels refer to the category
    if n == 6:
        labels = data["category"]
    # if there are 2 clusters, the labels refer to the type (A or Z)
    elif n == 2:
        labels = data["type"]
    # if there is another number of clusters, they cannot be associated to any specific label
    else:
        return sil, None, None, None, None

    # computing the rest of the metrics
    homo = metrics.homogeneity_score(labels, pred_labels)
    compl = metrics.completeness_score(labels, pred_labels)
    vm = metrics.v_measure_score(labels, pred_labels)
    rand = metrics.adjusted_rand_score(labels, pred_labels)
    return sil, homo, compl, vm, rand


def visualization(pred_labels, matrix, n):
    """
    Visualise metrics for each input representation
    5 scores for each possible result (2/6 clusters, token/tokens freq/tf-idf)
    Output: Print each score
    """
    # running the scores() function, and using the results here
    silhouette, homogeneity, completeness, v_measure, rand_index = scores(pred_labels, matrix, n)

    # printing all the results
    print("Intrinsic scores:")
    print("Silhouette coefficient: ", silhouette)
    print("Extrinsic scores:")
    print("Homogeneity: ", homogeneity)
    print("Completeness: ", completeness)
    print("V-measure: ", v_measure)
    print("Adjusted Rand index: ", rand_index)


# launch the whole program
if __name__ == "__main__":
    # opening the data to be used as input
    data = pd.read_csv('processed_data.csv', sep=',')
    # listing the 3 methods to be tested
    methods = ['tf-idf', 'token frequency', 'tokens']
    # listing the numbers of clusters to be tested
    clusters = [2, 6]
    # iterating over methods
    for m in methods:
        # for each method, iterating over the numbers of clusters
        for c in clusters:
            # displaying which method and the number of clusters used
            print(f'Clustering results using {c} clusters and method {m}')
            # launch the functions of the programm
            visualization(*clustering(data["processed_text"], c, m))
            # pass a lign
            print()
