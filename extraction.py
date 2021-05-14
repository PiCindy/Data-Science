import re
import nltk
import wptools
import wikipedia
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

def create_list(q, k):
    '''
    Creation of a list of k persons of category q.
    Input:
    q (str): category id
    k (int): nb of persons
    Output:
    List of k persons id of category q
    '''
    # Creating the SPARQL query
    query = "select distinct ?item where {?item wdt:P31 wd:Q5; wdt:P106 wd:%s.}" %(q)
    # Creating the Wrapper object
    sparql = SPARQLWrapper("http://query.wikidata.org/sparql", agent='sparqlwrapper 1.8.5 (rd-flib.github.io/sparqlwrapper)')
    sparql.setQuery(query)
    # Set the results format
    sparql.setReturnFormat(JSON)
    # Getting the results
    results = sparql.query().convert()
    # We keep the k+delta first results to be sure to have enough persons with all information needed
    return [result['item']['value'].split('/')[-1] for result in results["results"]["bindings"][:k+max(k//2,10)]]

def title_desc(identifier):
    '''
    Extraction of title and description of a person's page.
    Input:
    identifier (str): person id
    Output:
    Title of the page
    Description of the page
    '''
    # Opening the person's page
    page = wptools.page(wikibase=identifier)
    # Getting data about this page
    page.get_wikidata()
    # Returning title and description
    return page.data['title'], page.data['description']

def create_data(persons, category, k, n):
    '''
    Fills a list data with each person info
    Input:
    persons (list): list of persons
    category (str): category of the list
    k (int): number of persons per category
    n (int): number of sentences per person
    '''
    data = []
    # Finding the type of the category
    t = ''
    if category == 'singer' or category == 'writer' or category == 'painter':
        t = 'A'
    elif category == 'architect' or category == 'politician' or category == 'mathematician':
        t = 'Z'
    # Getting data for k persons
    i = 0
    while len(data) < k and i < len(persons):
        # Getting the current person
        p = persons[i]
        i+=1
        try:
            # Getting title and description of current person
            title, desc = title_desc(p)
            # Accessing the person's page
            page = wikipedia.page(title, auto_suggest=False)
            # Removing section names and line breaks in the summary
            summary = re.sub('==.+==', '', page.content).replace('\n', ' ')
            # Tokenizing sentences and converting it into a string
            sentences = ' '.join(nltk.sent_tokenize(summary)[:n])
            # Adding the list of person's info in data
            data.append([title, category, t, desc, sentences])
        # If an exception is found, we cannot have all needed elements and we ignore this person
        except wikipedia.exceptions.PageError:
            continue
        except LookupError:
            continue
        except wikipedia.DisambiguationError:
            continue
    return data


def extraction(k=30, n=5):
    '''
    Corpus extraction
    Parameters:
    k (int): number of persons per category - default value: 30
    n (int): number of sentences per person - default value: 5
    '''

    # Creation of the lists of k persons with category identifier
    singers = create_list("Q177220", k)
    writers = create_list("Q36180", k)
    painters = create_list("Q1028181", k)
    architects = create_list("Q42973", k)
    politicians = create_list("Q82955", k)
    mathematicians = create_list("Q170790", k)

    # Listing the categories
    variables = [singers, writers, painters, architects, politicians, mathematicians]
    categories = ['singer', 'writer', 'painter', 'architect', 'politician', 'mathematician']

    # Creating an empty list which will contain data
    data = []
    # Extending data for each category of persons
    for v, c in zip(variables, categories):
        data.extend(create_data(v, c, k, n))
    # Converting the list of lists into dataframe
    df = pd.DataFrame(data, columns = ['person', 'category', 'type', 'description', 'text'])
    # Storing the dataframe in a csv file
    df.to_csv('data/data.csv', index=False)


if __name__ == "__main__":
    # Running the extraction function
    extraction()
