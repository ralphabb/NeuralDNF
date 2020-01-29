# Neural#DNF
This is the code repository for the neural model counting system Neural#DNF, presented in the AAAI-20 [paper](https://arxiv.org/pdf/1904.02688.pdf) "Learning to Reason: Leveraging Neural Networks for Approximate Weighted Model Counting". It contains code for training and building the graph neural network, generating random DNF formulas with desired properties, and evaluating the network as described in the paper. It also contains the final training weights in the netParams_2 subdirectory.

## Requirements
- TensorFlow >=1.12 and the corresponding NumPy version
- JobLib >= 0.12.0 for data generation
- SciPy
- MatPlotLib for figure reproduction

## Datasets
The paper's training and test data sets are available for download in ZIP format [here](https://drive.google.com/open?id=1Xi-qJTxBJEXGYcsrZXisjJ2eDLBQRxSf).

## Running Neural#DNF

### Overall Experiments
To run overall experiments on a test dataset with thresholds 0.01, 0.02, 0.05, 0.1, 0.15, and 0.2, run 

```python runExperiments.py "../Data/TestSetDirectory/"``` 

Additional options include retrieving results by clause width, using `-widthBasedAnalysis T`, and changing the number of message passing iterations using `-numIter X`. A comprehensive list of options can be viewed using the command 

```python runExperiments.py -h```

### Experiments vs Formula Number of Variables
This test setting runs analogously to the earlier case, but instead reports performance versus number of formula variables *n*. This test can be run (additional options: `-h`) using the command:  

```python runExperimentsBySize.py "../Data/TestSetDirectory/"``` 

### Generating labelled DNF formulas
To generate data according to the paper's standard proportions, run  

```python generateData.py```

Additional options for generation are also provided, and a list of these can be found using the `-h` flag.
*Note:* Generation proportions can be changed by altering the three proportions arrays in the generateData.py script (dFS (distinct File Sizes), nCD (number (of) Clauses Distribution) and cWD (clause Width Distribution)).

### Training Neural#DNF
A single epoch of training on a training set can be run using the command

```python Train.py "../Data/TrainingSet/"```

The number of epochs can be changed using the `-nbTrainingEpochs X` flag, and further options can be found using `-h`

## Referencing this paper
If you make use of this code, or its accompanying [paper](https://arxiv.org/pdf/1904.02688.pdf), please cite this work as follows:

```
@inproceedings{ACL-AAAI20,
  title={Learning to {R}eason: Leveraging Neural Networks for Approximate {DNF} Counting},
  author    = {Ralph Abboud and
                {\.I}smail {\.I}lkan Ceylan and
               Thomas Lukasiewicz},
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2020}
}
```

