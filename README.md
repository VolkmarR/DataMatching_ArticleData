# Test project for data matching with article data

This repository contains python scripts to test the two data matching libraries 

* [Dedupe](https://github.com/dedupeio/dedupe)
* [Python Record Linkage Toolkit](https://github.com/J535D165/recordlinkage)

The scripts are developed using the use the PyCharm IDE and Python 3.6.

## Project structure

### Src

This directory contains the scripts.

* Tools.py (helper methods and classes)
* Evaluation.py (methods to evaluate data matching results)
* CompareMethods.py (script to test the various string comparison methods of the Python Record Linkage Toolkit)
* Dedupe.py (script to test the data matching using the Dedupe library)
* PythonRecordLinkageToolkit.py (script to test the data matching using the Python Record Linkage Toolkit library)

The test scripts need a config file passed as command line argument to run. Examples can be found in the Data directory.

### Data

This directory contains test data and test configuration files. 

The test data was downloaded from the [Database Group Leipzig](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution). 

## Reproducibility

The tests call the *init_random_with_seed* method to set the random number generator to a fixed seed value. 
Additionaly, the PYTHONHASHSEED environment variables must be set for the dedupe.py script. 
