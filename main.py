import extraction
import clustering
import classification

if __name__ == "__main__":
    # Data is already created, not necessary to run if it the parameters stay the same
    # Add parameters to change number of persons per category and number of sentences per description
    # extraction.extraction()
    # Preprocessed data is already created, not necessary to run it if the extraction didn't change
    # preprocessing.main()
    # Clustering
    clustering.main()
    # Classification
    classification.main()
