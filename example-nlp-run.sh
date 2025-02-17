#!/bin/sh

# Runs the eval process against two llama.cpp server backends, assuming an input csv at provided location

poetry run python ./nlp.py --type vandalism --threads 4 --backend llama-cpp-server --server-urls http://localhost:8080/v1,http://localhost:8081/v1 --input-csv ./osm_name_changes.csv 
