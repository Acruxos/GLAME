from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Literal, Optional
from urllib.error import HTTPError
from .sparql_util import get_neighbor_triples,get_property_label
import json
import time


def get_id_from_uri(uri: str) -> str:
    """Return the ID from a Wikidata URI."""
    return uri.split('/')[-1]

def init_triples_dict(subject, relation, obj) -> dict:
    """Initialize a dictionary to represent a triple."""
    return {
        "subject": subject,
        "relation": relation,
        "target": obj
    }


# Initialize the keys in the new json.
def initialize_json(json_obj: dict) -> dict:
    """Initialize the JSON object for the new triples."""
    return {
        "case_id": json_obj["case_id"],
        "triples": []
    }


# Get all the neighbor triples of an item as outgoing edges.
def get_entity_triplets(entity_id: str, entity_label: str) -> list:
    """Get the triples of an entity and its neighbors."""
    # Initialize a dictionary to store the mapping between entity URIs and their labels.
    entity_id_to_label = {}
    entity_id_to_label[entity_id] = entity_label

    # Get the object entities.
    results = get_neighbor_triples(entity_id)

    # First, process all the items with label information.
    for result in results["results"]["bindings"]:
        type = result['object']['type']
        if not type == "literal":
            continue
        subject = get_id_from_uri(result['subject']['value'])
        entity_id_to_label[subject] = result['object']['value']

    # Then, process all the edges (relations), and convert them to the required JSON format.
    triples = []
    for result in results["results"]["bindings"]:
        type = result['object']['type']
        if not type == "uri":
            continue

        subject = get_id_from_uri(result['subject']['value'])
        predicate = get_id_from_uri(result['predicate']['value'])
        obj = get_id_from_uri(result['object']['value'])

        triple = init_triples_dict(
            entity_id_to_label[subject],
            entity_id_to_label[predicate],
            entity_id_to_label[obj]
        )

        triples.append(triple)

    return triples


def get_triplets_from_dataset(dataset_path):
    """Get the triples from a dataset."""
    # Read the original dataset.
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)

    # Initialize a counter for storing the number of results.
    counter = 0
    relation_id_dict = {}
    new_json = []

    # Iterate over each case.
    for case in dataset:
        print("Processing Case{}".format(case['case_id']))

        # Initialize a new JSON object.
        new_data = initialize_json(case)

        # Iterate over each requested_rewrite.
        rewrite = case["requested_rewrite"]

        subject_label = rewrite["subject"]

        # Get the ID and label of the target_new.
        target_id = rewrite["target_new"]["id"]
        target_label = rewrite["target_new"]["str"]

        # Get the property label of the relation.
        if (rewrite["relation_id"] in relation_id_dict):
            relation_label = relation_id_dict[rewrite["relation_id"]]
        else:
            relation_label = get_property_label(rewrite["relation_id"])
            relation_id_dict[rewrite["relation_id"]] = relation_label

        # Get all the triples of the target_new.
        origin_triplet = init_triples_dict(
            subject_label, relation_label, target_label)
        target_neighbors = get_entity_triplets(
            target_id, target_label)

        # Add these triples to the new JSON object.
        new_data["triples"].extend(origin_triplet)
        new_data["triples"].extend(target_neighbors)

        counter += 1

        new_json.extend(new_data)

        # Decide the filename based on the counter.
        if counter % 1000 == 0:
            file_name = f'cf_graph_{counter}_to_{counter+999}.json'
            with open(file_name, 'w') as output_file:
                json.dump(new_json, output_file)
            new_json = []

    # If the last batch of data does not reach 1000 cases, save the remaining results to the final file.
    if counter % 1000 != 0:
        file_name = f'cf_graph_{counter - (counter % 1000)}_to_{counter - 1}.json'
        with open(file_name, 'w') as output_file:
            json.dump(new_json, output_file)
        new_json = []


if __name__ == "__main__":
    get_triplets_from_dataset(
        "/datasets/counterfact.json")
