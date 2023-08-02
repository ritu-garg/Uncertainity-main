## Getting Mahalanobis Scores for Out-of-Distribution samples detection

- Model is trained on CIFAR100 dataset
- CIFAR100 test dataset is used as in-distribution samples
- We used three Out-distribution samples datasets:


  - CIFAR 10
    ``` 
    python scores.py --ood_dataset cifar10
    ```
  - Fashion MNIST
    ```
    python scores.py --ood_dataset fashion_mnist
    ```
  - Mixup of CIFAR100 and CIFAR10
    ```
    python scores.py --ood_dataset mixup
    ```
