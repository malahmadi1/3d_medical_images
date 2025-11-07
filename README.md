## Installation
This code has been implemented in python language using Pytorch libarary and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:
* CentOS Linux release 7.3.1611
* Python 3.6.13
* CUDA 9.2
* PyTorch 1.9.0
* medpy 0.4.0
* tqdm，h5py

## Getting Started
Please download the prepared dataset from the following link and use the dataset path in the training and evalution code.
* [LA Dataset](https://drive.google.com/drive/folders/1_LObmdkxeERWZrAzXDOhOJ0ikNEm0l_l)
* [Pancreas Dataset](https://drive.google.com/drive/folders/1kQX8z34kF62ZF_1-DqFpIosB4zDThvPz)

Please change the database path and data partition file in the corresponding code.

### Training
To train the network on the LA dataset, execute python `pyhon train_LA`. For the Pancreas dataset, use python `pyhon train_pancreas`
### Evaluation
To evaluate the network on the LA dataset, run `pyhon test_LA`. For the Pancreas dataset, run `pyhon test_pancreas`







