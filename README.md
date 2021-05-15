Justine DILIBERTO,
Anna NIKIFOROVSKAJA,
Cindy PEREIRA

# M1 NLP Data Science project: Clustering and Classifying People based on Text and KB information
Collection of information about people belonging to different categories (singers, writers, painters, architects, politicians, mathematicians) and types (A for artists and Z for non-artists) using The Wikipedia online encyclopedia and The Wikidata knowledge base. Automatically clustering and classifying these people into the correct categories or types based on this information.

## Files

- main.py: main program to run
- extraction.py: program to extract information about people (Exercise 1)
- preprocessing.py: program to apply preprocessing methods on the descriptions and summaries about people (Exercise 2)
- clustering.py: program to compute clustering using different representation methods:
	- TFIDF
	- Token
	- Token-frequency

Each representation method is applied on two numbers of clusters:
	- 2 clusters (Types - A or Z)
	- 6 clusters (Categories - singers, writers, painters, architects, politicians, mathematicians)
- classification.py: program to compute classification using different algorithms:
	- Stochastic Gradient Descent Classifier
	- Support Vector Classifier
	- Multi-layer Perceptron Classifier

Each algorithm is applied on two kinds of information:
	- Types (A or Z)
	- Categories (singers, writers, painters, architects, politicians, mathematicians)

## Folders
- data: contains computed ready-to-use data files:
	- data.csv: raw data extracted by extraction.py
	- processed_data.csv: data processed by preprocessing.py (used for clustering and classification)

## Run the program

To run the program, launch main.py using the following optional arguments:
- \-\-parameters or -p followed by two integers corresponding to the number of people per category and the number of sentences per person. This is an optional argument as ready-to-use data is provided with the program (default values: 30 and 5).
> python3 main.py -p 10 3

**Warning**: The extraction of information may take a long time.

- \-\-classification or \-\-no-classification can be used to show the results of classification methods or to not run it. By default, it will run.
> python3 main.py --no-classification

- \-\-clustering or \-\-no-clustering can be used to show the results of clustering methods or to not run it. By default, it will run.
> python3 main.py --no-clustering

## Libraries used

- re
- nltk
- wptools
- wikipedia
- pandas
- SPARQLWrapper
- sklearn
- argparse
