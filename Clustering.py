from nltk import word_tokenize
from sklearn import metrics
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
        tfidf_matrix = tfidf_vectorizer.fit_transform(data)   # need to add data
        km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=5, verbose=0, random_state=3425)
        result = km.fit(tfidf_matrix)
        pred_labels = km.labels_
        return pred_labels, result

    # elif m == 'token freq':
    #

    # elif m == 'tokens':
    #


def scores():
    """
    Compute evaluation scores:
    Output: silhouette coeff, homogeneity, completeness, v-measure, adjusted Rand index
    """
    labels = []    # add original labels
    pred_labels = clustering(my_data, clusters, method)[0]
    matrix = clustering(my_data, clusters, method)[1]

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
    silhouette = scores()[0]
    homogeneity = scores()[1]
    completeness = scores()[2]
    v_measure = scores()[3]
    rand_index = scores()[4]
    print("Intrinsic scores:",
          "Silhouette coefficient: ", silhouette,
          "Extrinsic scores:",
          "Homogeneity: ", homogeneity,
          "Completeness: ", completeness,
          "V-measure: ", v_measure,
          "Adjusted Rand index: ", rand_index)



my_data = []
clusters = 2                     # input("Choose number of clusters: ")
method = 'tf-idf'                # input("Choose method (token, tokens freq, tf-idf): ")

clustering(my_data, clusters, method)
scores()
visualization()
