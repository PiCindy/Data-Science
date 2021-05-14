import re
import nltk
import string
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
    sparql = SPARQLWrapper("http://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # Getting the results
    results = sparql.query().convert()
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
    page = wptools.page(wikibase=identifier)
    page.get_wikidata()
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
    # Finding the type of the category
    data = []
    t = ''
    if category == 'singer' or category == 'writer' or category == 'painter':
        t = 'A'
    elif category == 'architect' or category == 'politician' or category == 'mathematician':
        t = 'Z'
    # Getting data for every person
    i = 0
    while len(data) < k and i < len(persons):
        p = persons[i]
        i+=1
        try:
            # Getting title and description
            title, desc = title_desc(p)
            # Accessing the person page
            page = wikipedia.page(title, auto_suggest=False)
            # Removing section names and line breaks in the summary
            summary = re.sub('==.+==', '', page.content).replace('\n', ' ')
            # Tokenizing sentences
            sentences = nltk.sent_tokenize(summary)[:n]
            # Adding a list with info in data
            data.append([title, category, t, desc, sentences])
        # If an exception is found, we cannot have all info and we ignore this person
        # Problem -> Not k persons in this category
        # Should replace this person by another one (create lists with more people?)
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
    k (int): number of persons per category
    n (int): number of sentences per person
    '''

    # Creation of the lists of persons
    singers = create_list("Q177220", k)
    writers = create_list("Q36180", k)
    painters = create_list("Q1028181", k)
    architects = create_list("Q42973", k)
    politicians = create_list("Q82955", k)
    mathematicians = create_list("Q170790", k)

    # Completion of data
    variables = [singers, writers, painters, architects, politicians, mathematicians]
    categories = ['singer', 'writer', 'painter', 'architect', 'politician', 'mathematician']

    data = []
    for v, c in zip(variables, categories):
        data.extend(create_data(v, c, k, n))
    data.to_csv('data.csv')


if __name__ == "__main__":
    extraction()
