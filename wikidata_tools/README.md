We use SPARQL queries to retrieve the subgraphs required for knowledge editing by sending HTTP requests to Wikidata server. The script `extract_subgraph.py` serves as a simple demonstration to crawl subgraphs required for each edit in the counterfact raw dataset and store them for editing purposes in the GLAME model.

## Usage

To use the `extract_subgraph.py` script, it is essential to adhere to the [Wikimedia User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy). Wikimedia sites require an HTTP User-Agent header for all requests, which necessitates users to fill in the User-Agent within the script as a distinct identifier and include their contact information within its content. For further details, please refer to the original policy content.

1. Fill in the path to the dataset in `extract_subgraph`, and the User-Agent information as per the Wikimedia User-Agent policy in `sparql_util.py`.
2. Run the `extract_subgraph.py` script to crawl the subgraphs required for editing each entry in the dataset.