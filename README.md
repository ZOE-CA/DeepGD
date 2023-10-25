# DeepGD
This repository is a companion page for the following paper:  “DeepGD: A Multi-Objective Black-Box Test Selection Approach for Deep Neural Networks”, (submitted paper).
This paper is implemented in python on Google Colab .

This repository is a replication package of DeepGD, a black-box (BB) test selection approach for deep neural networks (DNNs) that uses a customized multi-objective genetic search to guide the selection of test inputs with high fault-revealing power. It relies on diversity and uncertainty scores and it only requires 
access to the complete test dataset and prediction probabilities of the DNN model. 
The approach also considers a clustering-based approach to estimate faults in DNN models. 
An empirical evaluation on five DNN models, four datasets, and nine baselines shows that DeepGD provides better guidance for selecting 
inputs with high fault-revealing power and improves model performance through retraining.


Our main contributions are:

1- Proposing BB test selection approach

2- Customized the multi-objective search and validating by doing an ablation study

3- Validating DeepGD by approximating faults in DNNs

4- Comparing existing test selection metrics with DeepGD in terms of fault detection abilities, execution time, diversity, and retraining improvement 


* [Baseline_results](Baseline_results/) folder contains a selected subsets by each methods through different datasets and models.
* [Fault_clusters](Fault_clusters/) folder contains DNNs' faults which are saved in it for six different combinations of models & datasets.
* [Retraining](Retraining/) folder contains the retrained models with the selected subsets of all methods.
* [LSA_DSA](LSA_DSA/) folder contains some parts of [4] for computing the LSA and DSA scores.
* [ATS-master_final](ATS-master_final/) folder contains the related code for applying ATS from [3].

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
     |
     |---  Generating data               Our implementation for generating test inputs by applying real transformations.
     
     
  

Research Questions
---------------
Our experimental evaluation answers the research questions below.

_**RQ1: Do we find more faults than existing test selection approaches with the same testing budget?**_

_**RQ2:  Do we more effectively guide the retraining of DNN
models with our selected inputs than with baselines?**_

_**RQ3. Can DeepGD select more diverse test input sets compared to other test selection approaches?**_

_**RQ4. How do DeepGD and baseline approaches compare in terms of computation time?**_


Not in the paper:
-----
In our study, we would like to emphasize that the use of mispredictions as the sole criterion for evaluating test selection approaches can be misleading and may not accurately reflect the fault detection capabilities of these methods. This is primarily due to the presence of redundant test inputs from a fault prediction standpoint. Therefore, the specific numbers related to mispredicted inputs are not included in the main text of our paper. Instead, they are provided within this replication package, along with other supplementary data.

_**RQ5.How do DeepGD and baseline approaches compare in terms of the number of selected mispredicted inputs?**_
This research question seeks to determine and quantify the differences between DeepGD and baseline approaches concerning their ability to identify and utilize test inputs that lead to mispredictions.
This inquiry is prompted by one of the comments received during the review process

![image](https://github.com/ZOE-CA/DeepGD/assets/109688199/cdcbc3b1-dc47-418a-8ba5-196100d156f0)



Notes
-----

1- To speed-up the execution, you can use GPU-based TensorFlow by changing the Colab Runtime.

References
-----
1- [DeepGini](https://dl.acm.org/doi/abs/10.1145/3395363.3397357)

2- [Diversity](https://www.researchgate.net/publication/357301807_Black-Box_Testing_of_Deep_Neural_Networks_through_Test_Case_Diversity)

3- [ATS](https://conf.researchr.org/details/icse-2022/icse-2022-papers/184/Adaptive-Test-Selection-for-Deep-Neural-Networks)

4- [Surprise Adequacy](https://github.com/coinse/sadl)

5- [Black-box Safety Analysis and Retraining of DNNs based on Feature Extraction and Clustering](https://www.semanticscholar.org/paper/Black-box-Safety-Analysis-and-Retraining-of-DNNs-on-Attaoui-Fahmy/a29c208751555a4c2d4874070b8555fc53e5a414)
