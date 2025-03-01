# Reasoning-Grounded Natural Language Explanations

## Structure of the repository

* *explainable_llms* - Python package with the logic
* *scripts* - Scripts for running the experiments
* *tests* - Pytest tests

## How to set up

1. Install [Poetry](https://python-poetry.org/)
2. Clone this repository and `cd` into it
3. Install the dependencies using the command `poetry install`
4. Activate the newly created virtual environment using `poetry shell`
5. The experiments can be executed using the scripts located in the `scripts` folder

## Datasets

* [Logistic regression datasets](https://drive.google.com/file/d/1reU1v5DQGVMhFS8Or1RW87rdTwrGr6jK/view?usp=sharing) (in the paper, n_inputs_8/dataset_0 was used)
* [Decision tree datasets](https://drive.google.com/file/d/16qJHCcmJJMQXNky059h__Hxmz8vogKWr/view?usp=sharing) (in the paper, n_inputs_2_depth_6/dataset_0 was used; the actual tree depths are deeper by 1 than what is stated in the dataset name)
* [Natural language decision tree dataset](https://drive.google.com/file/d/1teOGNq8_wfcC0FhDtnQ0sZC7HkAWrqb8/view?usp=sharing)
