import pandas as pd
from nltk import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering


def clustering(data, n, m):
    """Clustering algorithm
    Input:
    data (Series): serie of preprocessed texts
    n (int): number of clusters
    m (str): representation method
    Output:
    Predicted labels of clusters, matrix of clusters, number of clusters
    """

    # First case: tf-idf method
    if m == 'tf-idf':
        # Creating the TFIDF vectorizer
        vectorizer = TfidfVectorizer(max_features=8000, # We want 8000 features
                                           use_idf=True,
                                           stop_words='english', # The stop words to be removed
                                           tokenizer=word_tokenize, # The way of tokenizing
                                           ngram_range=(1, 3))

    # Second case: token frequency method
    elif m == 'token frequency':
        # Creating the Token frequency vectorizer
        vectorizer = CountVectorizer(binary=False)

    # Third case: tokens method
    elif m == 'tokens':
        # Creating the Tokens vectorizer
        vectorizer = CountVectorizer(binary=True)

    # Fitting the model with data
    x_count = vectorizer.fit_transform(data)
    # Converting the sparse matrix into matrix
    matrix = x_count.todense()

    # After having tried with KMeans, we observed better results for Agglomerative clustering
    #km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=30, verbose=0, random_state=3425)
    #km.fit(matrix)

    # Using the agglomerative clustering algorithm
    ac = AgglomerativeClustering(n_clusters=n, affinity='euclidean', memory=None, connectivity=None,
                                 compute_full_tree='auto', linkage='ward', distance_threshold=None,
                                 compute_distances=False)
    # Fitting the matrix
    ac.fit(matrix)
    # Renaming the labels
    pred_labels = ac.labels_
    return pred_labels, matrix, n


def scores(data, pred_labels, matrix, n):
    """
    Compute evaluation scores
    Output: silhouette coeff, homogeneity, completeness, v-measure, adjusted Rand index
    """

    # Calling this score first, as it does not depend on the true labels
    sil = metrics.silhouette_score(matrix, pred_labels, sample_size=1000)

    # If there are 6 clusters, the labels refer to the category
    if n == 6:
        labels = data["category"]
    # If there are 2 clusters, the labels refer to the type (A or Z)
    elif n == 2:
        labels = data["type"]
    # If there is another number of clusters, they cannot be associated to any specific label
    else:
        return sil, None, None, None, None

    # Computing the rest of the metrics
    homo = metrics.homogeneity_score(labels, pred_labels)
    compl = metrics.completeness_score(labels, pred_labels)
    vm = metrics.v_measure_score(labels, pred_labels)
    rand = metrics.adjusted_rand_score(labels, pred_labels)
    return sil, homo, compl, vm, rand


def visualization(data, pred_labels, matrix, n):
    """
    Visualise metrics for each input representation
    5 scores for each possible result (2/6 clusters, token/tokens freq/tf-idf)
    Output: Print each score
    """
    # Running the scores() function, and storing the results
    silhouette, homogeneity, completeness, v_measure, rand_index = scores(data, pred_labels, matrix, n)

    # Printing all the results
    print("Intrinsic scores:")
    print("Silhouette coefficient:", silhouette)
    print("Extrinsic scores:")
    print("Homogeneity:", homogeneity)
    print("Completeness:", completeness)
    print("V-measure:", v_measure)
    print("Adjusted Rand index:", rand_index)

def main(data):
    # Listing the 3 methods to be tested
    methods = ['tf-idf', 'token frequency', 'tokens']
    # Listing the numbers of clusters to be tested
    clusters = [2, 6]
    # Iterating over methods
    for m in methods:
        # For each method, iterating over the numbers of clusters
        for c in clusters:
            # Displaying which method and the number of clusters are used
            print(f'Clustering results using {c} clusters and method {m}')
            # Running the clustering and visualization functions
            visualization(data, *clustering(data["processed_text"], c, m))
            # Print a blank line to separate different tests
            print()


# Launch the whole program
if __name__ == "__main__":# Importing the data to be used as input
    data = pd.read_csv('data/processed_data.csv', sep=',')
    main(data)
