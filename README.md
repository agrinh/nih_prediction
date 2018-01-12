# NIH Prediction

## Introduction
Example implementation illustrating how one can read the NIH chest xray data
using the TensorFlow Datasets API. It does this by showing how to train a
ResNet v2 152 to predict the disieses as a multilabel prediction problem. To
make the scenario more interesting it also shows how to split the training
across a number of GPUs simultaneously by using a data parallel or towering
approach. Though the intent here isn't to explore the dataset it is quickly walked through in the notebook `notebooks/nih_read_scans.ipynb`.

The data is freely available from both NIH and Kaggle at the following links:
* https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
* https://www.kaggle.com/nih-chest-xrays/data

A Dockerfile is provided that sets up a simple instance with all dependencies
installed to run the code.

Finally a short presentation on the TensorFlow Datasets API is included in this
repo. It is meant to be presented in conjunction with the Jupyter notebooks.

## Installation
For Python3 and assuming the current working directory is the repo root

__Local installation__
```bash
pip install -r requirements.txt
```

__Docker install__ (requires `nvidia-docker`)
```bash
nvidia-docker build . -t nih_predict
```
(to start use `nvidia-docker run` and make sure to map up directories for data
and code)

## Usage

### Notebooks
The notebooks show a very simple implementation and have all code inline to
easily illustrate how to use the features explored here. For a slightly deeper
exploration and to show a more complete example see the package nih_prediction.

### Compute mean and std of scans
`nih_prediction/compute_statistics.py` may be used to predict the mean and std
of the scans. The computed values have been set as default in the train script.
Note that one should ideally only do this on the training dataset. Note that
the PYTHONPATH must be set to include the source directory `nih_prediction`.

### Train model
To train the model point the training script to the path of the expanded NIH
Chest Xray data. Note that images should be expanded to a subdirectory called
images. To run on 4 K80 GPUs on a system with 32 CPU cores e.g.:

```
PYTHONPATH=. python3 nih_prediction/train.py \
    --num-workers 32 \
    --batch-size 16 \
    --gpus 0,1,2,3 \
    /path/to/expanded/nih/data
```


### Contributions are welcome!
