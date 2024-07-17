from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Literal, Optional
from urllib.error import HTTPError
import json
import time


# Wikidata query service setting
wikidata_url = "https://query.wikidata.org/sparql"
user_agent = "" # HTTP User-Agent header
sparql = SPARQLWrapper(wikidata_url, agent=user_agent)


def safe_sparql_request(sparql):
    """Make a SPARQL request with error handling."""
    while True:
        try:
            # Requests that make too many calls may result in a 429 error.
            results = sparql.query().convert()
            return results

        except HTTPError as e:
            if e.code == 429:
                print("429 error occurred. Pausing execution for a while...")
                time.sleep(15)  # Sleep for 15 seconds before retrying.
            else:
                # If the error code is not 429, re-raise the exception.
                raise e


# According to the property id, get the label of the relation.
def get_property_label(property_id: str) -> str:
    """Get the label of a property given its ID."""
    queryString = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd: <http://www.bigdata.com/rdf#>

    SELECT ?propertyLabel WHERE {
    wd:%s rdfs:label ?propertyLabel.
    FILTER(LANG(?propertyLabel) = "en")
    }
    """ % property_id

    sparql.setQuery(queryString)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    return results['results']['bindings'][0]['propertyLabel']['value']


# According to the item id, get the neighbor triples of the entity.
def get_neighbor_triples(item_id: str) -> str:
    """Get the triples of the neighbors of an entity given its ID."""
    queryString = """
    CONSTRUCT {
      ?s ?p ?o .
      ?p rdfs:label ?pLabel .
      ?o rdfs:label ?oLabel .
    } 
    WHERE { 
      BIND(wd:%s AS ?s) 
      ?s ?p ?o .
      ?wdProp wikibase:directClaim ?p .
      
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
        ?wdProp rdfs:label ?pLabel .
        ?o rdfs:label ?oLabel .
      }

      FILTER(STRSTARTS(STR(?p), STR(wdt:))) .
      FILTER(STRSTARTS(STR(?o), STR(wd:))) .
      FILTER(!STRSTARTS(STR(?p), STR(wdt:P910))) .
      FILTER(!STRSTARTS(STR(?p), STR(wdt:P1423))) .
    } 
    """ % item_id

    sparql.setQuery(queryString)

    sparql.setReturnFormat(JSON)
    label = sparql.query().convert()

    return label