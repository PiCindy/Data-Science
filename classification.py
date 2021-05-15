import nltk
import pandas as pd

import sklearn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer # import tf-idf vectorizer from sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_classification_results(y_gold, y_pred):
    """
    Computes classification results, such as accuracy, precision, recall, F1 score, confusion matrix
    using true values and predicted ones.
    Input:
    y_gold (Array-like of ints): true values from the dataset
    y_pred (Array-like of ints): predicted values from the algorithm
    Output:
    Accuracy, confusion matrix and classification report (contains precision, recall and F1 score)
    """
    # Computing the accuracy score
    accuracy = accuracy_score(y_gold, y_pred)
    # Computing the confusion matrix
    conf_matrix = confusion_matrix(y_gold, y_pred)
    # Computing the classification report
    classification_rep = classification_report(y_gold, y_pred)
    # Returning the results
    return accuracy, conf_matrix, classification_rep

def visualize_classification_results(name, tested_on, y_gold, y_pred):
    """
    Prints classification results, such as accuracy, precision, recall, F1 score, confusion matrix
    using true values and predicted ones.
    Input:
    name (String): name of the algorithm tested
    tested_on (String): what was the goal of prediction (categories, types, etc)
    y_gold (Array-like of ints): true values from the dataset
    y_pred (Array-like of ints): predicted values from the algorithm
    Output:
    accuracy, confusion matrix and classification report (contains precision, recall and F1 score)
    """
    # Displaying which algorithm was used and the goal
    print(f"Classification results of testing {name} on {tested_on}.\n")

    # Computing results
    acc, conf_matrix, class_rep = get_classification_results(y_gold, y_pred)
    # Printing the results
    print("Accuracy:", acc)
    print("Confusion matrix:\n", conf_matrix)
    print("Classification report:\n", class_rep)
    print("\n")

def run_classification(data):
    """
    Runs classification on several algorithms, also prints the results.
    Stratification is used on data so that the train and test datasets were balanced based on categories.
    Input:
    data (Dataframe): dataframe with labels "processed_text", "category" and "type"
    """

    # Creating the TFIDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=8000, # We want 8000 features
                                       use_idf=True,
                                       stop_words='english', # The stop words to be removed
                                       tokenizer=nltk.word_tokenize, # The way of tokenizing
                                       ngram_range=(1, 3))
    # Fitting the model with data
    X_tfidf = tfidf_vectorizer.fit_transform(data["processed_text"])
    # Extracting y
    y_cat = data['category']
    y_type = data['type']
    # Splitting data into train and test samples
    X_train, X_test, y_cat_train, y_cat_test, y_type_train, y_type_test = train_test_split(X_tfidf, y_cat, y_type,
                                                                                           test_size=0.3, stratify=y_cat)
    # Listing different algorithms used and creating the models
    classification_algos = [SGDClassifier(max_iter=1000, tol=1e-3),
                           SVC(gamma='auto'),
                           MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1)]
    # Fitting the models, predicting labels, and visualizing results for each algorithm
    for algo in classification_algos:
        # Fitting the model with categories
        algo.fit(X_train, y_cat_train)
        # Predicting the category
        y_pred = algo.predict(X_test)
        # Visualizing results
        visualize_classification_results(algo.__class__.__name__, "categories", y_cat_test, y_pred)
        # Fitting the model with types
        algo.fit(X_train, y_type_train)
        # Predicting the type
        y_pred = algo.predict(X_test)
        # Visualizing results
        visualize_classification_results(algo.__class__.__name__, "types", y_type_test, y_pred)

def main(data):
    # Running the classification method
    run_classification(data)

if __name__ == "__main__":
    # Importing the data to be used as input
    data = pd.read_csv('data/processed_data.csv')
    main(data)
