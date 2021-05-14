import pandas as pd
from nltk import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def clustering(data, n, m):
    """Clustering algorithm
    Input:
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
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)
        km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=5, verbose=0, random_state=3425)
        matrix = km.fit(tfidf_matrix)
        pred_labels = km.labels_
        return pred_labels, matrix, n

    elif m == 'token frequency':
        vectorizer = CountVectorizer(binary=False)
        x_count = vectorizer.fit_transform(data)
        token_freq_matrix = x_count.todense()
        km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=5, verbose=0, random_state=3425)
        matrix = km.fit(token_freq_matrix)
        pred_labels = km.labels_
        return pred_labels, matrix, n

    elif m == 'tokens':
        vectorizer = CountVectorizer(binary=True)
        x_count = vectorizer.fit_transform(data)
        token_freq_matrix = x_count.todense()
        km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=5, verbose=0, random_state=3425)
        matrix = km.fit(token_freq_matrix)
        pred_labels = km.labels_
        return pred_labels, matrix, n


def scores():
    """
    Compute evaluation scores:
    Output: silhouette coeff, homogeneity, completeness, v-measure, adjusted Rand index
    """
    pred_labels, matrix, n = clustering(text, clusters, method)

    if n == 6:
        labels = my_data["category"]
        sil = metrics.silhouette_score(matrix, pred_labels, sample_size=1000)
        homo = metrics.homogeneity_score(labels, pred_labels)
        compl = metrics.completeness_score(labels, pred_labels)
        vm = metrics.v_measure_score(labels, pred_labels)
        rand = metrics.adjusted_rand_score(labels, pred_labels)
        return sil, homo, compl, vm, rand

    elif n == 2:
        labels = my_data["type"]
        sil = metrics.silhouette_score(matrix, pred_labels, sample_size=1000)
        homo = metrics.homogeneity_score(labels, pred_labels)
        compl = metrics.completeness_score(labels, pred_labels)
        vm = metrics.v_measure_score(labels, pred_labels)
        rand = metrics.adjusted_rand_score(labels, pred_labels)
        return sil, homo, compl, vm, rand


def visualization():
    """
    Visualise metrics for each input representation
    5 scores for each possible result (2/6 clusters, token/tokens freq/tf-idf)
    Output: Print each score
    """
    silhouette, homogeneity, completeness, v_measure, rand_index = scores()

    print("Intrinsic scores:",
          "Silhouette coefficient: ", silhouette,
          "Extrinsic scores:",
          "Homogeneity: ", homogeneity,
          "Completeness: ", completeness,
          "V-measure: ", v_measure,
          "Adjusted Rand index: ", rand_index)


my_data = pd.read_csv("processed_data.csv", sep=',')
text = list(my_data["processed_text"])


clusters = 2                     # choose number of clusters here
method = 'tf-idf'                # choose method here ('tf-idf', 'token frequency', 'tokens')

clustering(text, clusters, method)
scores()
visualization()
