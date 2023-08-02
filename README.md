# Uncertainity scores using Mahalanobis Distance

- The code is developed using Python 3.10.6

## Requirements

- Install pytorch 1.13.1 following the instructions on the [official website](https://pytorch.org/).

- Then install the other dependencies.

```
pip install -r requirements.txt

```

## Dataset

- Used CIFAR100 , CIFAR10 datasets from tensorflow datasets

## Model

- Used Pre-trained Vision Transformer model from transformers library

## Running the code

- Train the Vision Transformer model on CIFAR100 dataset.

```
cd Code
sh run.sh
```

- After training the model will be saved in `Models` folder

- Get the Mahalanobis Scores.

```
cd Code
python scores.py

```





