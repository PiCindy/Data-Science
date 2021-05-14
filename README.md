Justine DILIBERTO,
Anna NIKIFOROVSKAYA,
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

# Needs modifications

To run the program, launch main.py using (for example):
> python3 main.py

In main.py, the line 8 is commented. It is not necessary to run the extraction if you want to keep the basic data (with 30 persons per category and 5 sentences per person). If needed, it is possible to re-extract this basic data without providing any argument by uncommenting this line:
> l.8 extraction.extraction()

If you want to change these parameters, you can run it this way:
> l.8 extraction.extraction(nb_of_persons_per_category, nb_of_sentences_per_person)

The line 10 is commented. If you kept the basic data, the preprocessed data is already created. If you changed the extraction parameters, you need to run the preprocessing program by uncommenting this line:
> l.10 preprocessing.main()

## Libraries used

- re
- nltk
- wptools
- wikipedia
- pandas
- SPARQLWrapper
- sklearn
