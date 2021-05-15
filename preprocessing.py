import nltk
import pandas as pd
from nltk.corpus import stopwords

def preprocessing(text):
    """
    Application of all preprocessing methods on the text.
    Input:
    text (string): a Wikipedia summary or Wikidata description
    Output:
    processed: the text after preprocessing
    """
    # Tokenize the text
    processed = nltk.word_tokenize(text)
    # Lowercase the tokens
    processed = [token.lower() for token in processed]
    # Remove stop words
    en_stopwords = stopwords.words('english')
    processed = [token for token in processed if token not in en_stopwords]
    # Returns the string of tokens
    return ' '.join(processed)

def main(df):
    # Creating a new dataframe
    data = pd.DataFrame()
    # Creating the dataframe
    data['person'] = df['person']
    data['text'] = df['text']
    # Apply preprocessing methods to texts
    data['processed_text'] = df['text'].apply(preprocessing)
    data['description'] = df['description']
    # Apply preprocessing methods to descriptions
    data['processed_description'] = df['description'].apply(preprocessing)
    data['type'] = df['type']
    data['category'] = df['category']
    # Storing the processed data in a csv file
    data.to_csv('data/processed_data.csv', index=False)

if __name__ == "__main__":
    # Importing data
    df = pd.read_csv('data/data.csv', sep=',')
    main(df)
