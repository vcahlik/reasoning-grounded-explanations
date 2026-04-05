# Reasoning-Grounded Natural Language Explanations for Language Models

This repository contains the code associated with the research paper **Reasoning-Grounded Natural Language Explanations for Language Models** ([website](https://cahlik.net/reasoning-grounded-explanations-paper/), [arXiv](https://arxiv.org/abs/2503.11248), [PDF](https://arxiv.org/pdf/2503.11248)).

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

We recommend downloading the associated datasets from our [Zenodo repository](https://zenodo.org/uploads/15149693). Alternatively, you may use the following Google Drive links:

* [Logistic regression datasets](https://drive.google.com/file/d/1reU1v5DQGVMhFS8Or1RW87rdTwrGr6jK/view?usp=sharing) (in the paper, n_inputs_8/dataset_0 was used)
* [Decision tree datasets](https://drive.google.com/file/d/16qJHCcmJJMQXNky059h__Hxmz8vogKWr/view?usp=sharing) (in the paper, n_inputs_2_depth_6/dataset_0 was used; the actual tree depths are deeper by 1 than what is stated in the dataset name)
* [Natural language decision tree dataset](https://drive.google.com/file/d/1teOGNq8_wfcC0FhDtnQ0sZC7HkAWrqb8/view?usp=sharing)

## Citing

In the case that you find this repository or datasets helpful, feel free to cite the [published paper](https://doi.org/10.1007/978-3-032-08327-2_1):

```
@inproceedings{cahlik2025reasoning,
  title={Reasoning-grounded natural language explanations for language models},
  author={Cahlik, Vojtech and Alves, Rodrigo and Kordik, Pavel},
  booktitle={World Conference on Explainable Artificial Intelligence},
  pages={3--18},
  year={2025},
  organization={Springer}
}
```
