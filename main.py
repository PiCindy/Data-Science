import nltk
#import extraction
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
    return ' '.join(processed)

df = pd.read_csv('data.csv')
data = pd.DataFrame()
data['person'] = df['person']
data['text'] = df['text']
data['processed_text'] = df['text'].apply(preprocessing)
data['description'] = df['description']
data['processed_description'] = df['description'].apply(preprocessing)
data['category'] = df['category']
data['type'] = df['type']
data.to_csv('processed_data.csv')
