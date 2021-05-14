import pandas as pd
from nltk import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def clustering(data, n, m):
    """Clustering algorithm
    Input:
    data ():
    n (int): number of clusters
    m (str): representation method
    Output:
    Predicted labels of clusters, matrix of clusters
    """

    if m == 'tf-idf':
        tfidf_vectorizer = TfidfVectorizer(max_features=8000,
                                           use_idf=True,
                                           stop_words='english',
                                           tokenizer=word_tokenize,
                                           ngram_range=(1, 3))
        matrix = tfidf_vectorizer.fit_transform(data).todense()

    elif m == 'token frequency':
        vectorizer = CountVectorizer(binary=False)
        x_count = vectorizer.fit_transform(data)
        matrix = x_count.todense()

    elif m == 'tokens':
        vectorizer = CountVectorizer(binary=True)
        x_count = vectorizer.fit_transform(data)
        matrix = x_count.todense()

    #km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=30, verbose=0, random_state=3425)
    #km.fit(matrix)
    ac = AgglomerativeClustering(n_clusters=n, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False)
    ac.fit(matrix)
    pred_labels = ac.labels_
    return pred_labels, matrix, n

def scores(pred_labels, matrix, n):
    """
    Compute evaluation scores:
    Output: silhouette coeff, homogeneity, completeness, v-measure, adjusted Rand index
    """

    sil = metrics.silhouette_score(matrix, pred_labels, sample_size=1000)

    if n == 6:
        labels = data["category"]
    elif n == 2:
        labels = data["type"]
    else:
        return sil, None, None, None, None

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
    silhouette, homogeneity, completeness, v_measure, rand_index = scores(pred_labels, matrix, n)

    print("Intrinsic scores:")
    print("Silhouette coefficient: ", silhouette)
    print("Extrinsic scores:")
    print("Homogeneity: ", homogeneity)
    print("Completeness: ", completeness)
    print("V-measure: ", v_measure)
    print("Adjusted Rand index: ", rand_index)


if __name__ == "__main__":
    data = pd.read_csv('processed_data.csv', sep=',')
    methods = ['tf-idf', 'token frequency', 'tokens']
    clusters = [2, 6]
    for m in methods:
        for c in clusters:
            print(f'Clustering results using {c} clusters and method {m}')
            visualization(*clustering(data["processed_text"], c, m))
            print()
