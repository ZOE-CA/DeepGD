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


* [LSA_DSA](LSA_DSA/) folder contains some parts of [1] for computing the LSA and DSA scores.
* [ATS-master_final](ATS-master_final/) folder contains the related code for applying ATS from [].
* [Baseline_results](Baseline_results/) folder contains some parts of [1] for computing the LSA and DSA scores.
* [Fault_cluster](Fault_cluster/) folder contains some parts of [1] for computing the LSA and DSA scores.
Requirements
---------------
You need to first install these Libraries:
  - `!pip install umap-learn`
  - `!pip install tslearn`
  - `!pip install hdbscan`

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

1. First, you need to upload the repo on your Google drive and run the codes on [Google Colab](https://colab.research.google.com)
2. The main code that you need to run is ``. This code covers all the datasets and models that we used in the paper, 
however if you want to test the code on ther models and datasets which are not used in our paper, you need to change two lines of the code in `` that are related to the loading model and the selected layer. 
To do so please:

Change these lines :

`model= load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet1.h5")`




Repository Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    Replication-package
     .
     |
     |---                     Pre-trained models used in the paper (LeNet-1, LeNet-4, LeNet-5, 12-Layer ConvNet, ResNet20)
     |
     |---              Random samples (60 subsets with sizes of 100,...,1000) to replicate the paper's results
     |
     |---                            The preprocessing time related to VGG feature extaction on MNIST dataset             
  

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
