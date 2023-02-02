# DeepGD

This repository is a companion page for the DeepGD paper 

This paper is implemented in python language with GoogleColab (It is an open-source and Jupyter based environment).

DeepGD is a black-box test selection approach for deep neural networks (DNNs) that uses a customized multi-objective genetic search to guide
the selection of test inputs with high fault-revealing power. It relies on diversity and uncertainty scores and it only requires 
access to the complete test dataset and prediction probabilities of the DNN model. 
The approach also considers a clustering-based approach to estimate faults
in DNN models. An empirical evaluation on five DNN models, four datasets, and nine baselines shows that DeepGD provides better guidance for selecting 
inputs with high fault-revealing power and improves model performance through retraining.


Our main contributions are:

1- Proposing BB test selection approach

2- Customized the multi-objective search

3- Validating a DeepGD by approximating faults in DNNs

4- Comparing existing test selection metrics with DeepGD in terms of fault detection abilities and retraining improvement 


* [Baseline_results](Baseline_results/) folder contains a selected subsets by each methods through different datasets and models.
* [Fault_clusters](Fault_clusters/) folder contains DNNs' faults which are saved in it for six different combinations of models & datasets.
* [Retraining](Retraining/) folder contains the retrained models with the selected subsets of all methods.
* [LSA_DSA](LSA_DSA/) folder contains some parts of [1] for computing the LSA and DSA scores.
* [ATS-master_final](ATS-master_final/) folder contains the related code for applying ATS from [].

Requirements
---------------
You need to first install these Libraries:
  - `!pip install umap-learn`
  - `!pip install tslearn`
  - `!pip install hdbscan`
  - `!pip install pymoo`

The code was developed and tested based on the following environment:

- python 3.8
- keras 2.7.0
- Tensorflow 2.7.1
- pytorch 1.10.0
- torchvision 0.11.1
- matplotlib
- sklearn

---------------
Here is a documentation on how to use this replication package.

### Getting started

To run the code, you have to upload the repository to Google Drive and open it on Google Colab and just the first line and set your local path and run the code.





Repository Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    Replication-package
     .
     |
     |---  Final DeepGD                  Customized multi-objective search-based test selection method
     |
     |---  BaseLine methods              All the used test selection baselines that have used in the paper.
     |
     |---  Retraining                    The RQ2 experiment.
     
  

Research Questions
---------------
Our experimental evaluation answers the research questions below.

_**1- RQ1: ?**_

Notes
-----

1- We used the same recommended settings of LSC and DSC hyperparameters (upper bound, lower bound, number of buckets, etc.) as in the original paper for the different models and datasets in our experiments.

2- For speed-up, you can use GPU-based TensorFlow by changing the Colab Runtime.

References
-----
1- [Surprise Adequacy](https://github.com/coinse/sadl)

3- [Revisiting Neuron Coverage Metrics and Quality of Deep Neural Networks](https://github.com/soarsmu/Revisiting_Neuron_Coverage/blob/master/Correlation/coverage.py)

4- [Supporting deep neural network safety analysis and retraining](https://www.researchgate.net/publication/339015259_Supporting_DNN_Safety_Analysis_and_Retraining_through_Heatmap-based_Unsupervised_Learning)

5- [Black-box Safety Analysis and Retraining of DNNs based on Feature Extraction and Clustering](https://www.semanticscholar.org/paper/Black-box-Safety-Analysis-and-Retraining-of-DNNs-on-Attaoui-Fahmy/a29c208751555a4c2d4874070b8555fc53e5a414)
