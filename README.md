# Recovering-Label-from-Update
This is the official implementation of the ICML2024 paper "Recovering Labels from Local Updates in Federated Learning".

## Requirements
This code is implemented in Pytorch and Cuda. We ran all experiments in the virtual environment created by Conda.
For installing the virtual environment:
```
conda create -n RLU python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Dataset Preparing:
* Create directory `mkdir data` and subdirectory `mkdir data/data_partitions`
* **SVHN/CIFAR10/CIFAR100/MNIST**: should automatically be downloaded via PyTorch datasets.
* **TinyImageNet**: download dataset from curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip.
  

After unzipping the downloaded files, place them in the `data/` directory.

> 
## Project Structure
* `option.py`: this file contains configuartions for all schemes. 
* `DataSampling.py`: this file is relevant to data partions for federated learning
* `models.py`: this file define models for different experiments. Please change the activation functions if needed.
* `utils.py`: general utilization 
* `llg.py`: this is adapted from the attack scheme [LLG](https://github.com/tklab-tud/LLG)
* **./client**: for each method, we defined adaptive client objects. We currently only release FedAvg and FedProx. FedDyn and Scaffold will be soon.
* `./client/client_utils.py`: utilization for the attacking schemes


## Run Attacks:

#### For SVHN with multiple local epochs with FedAvg

```
python main.py  --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model resnet18 --hidden 512 --n_classes 10 --scheme VLI
python main.py --scheme iRLG --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme RLU --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme LLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme ZLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
```

#### For CIFAR10 with multiple local epochs with FedAvg

```
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
python main.py --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512 --scheme RLU 
python main.py  --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model mobilenetv2 --hidden 512 --scheme All
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
```

#### For CIFAR100 with multiple local epochs with FedAvg
```
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py  --local_epoch 10 --dataset CIFAR100 --n_classes 200 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512 --scheme RLU
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
```

#### For SVHN with multiple local epochs with FedProx

```
python main.py --scheme iRLG --local_epoch 10 --dataset SVHN --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme RLU --local_epoch 10 --dataset SVHN --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme LLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme ZLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
```

#### For CIFAR10 with multiple local epochs with FedProx

```
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
python main.py --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --fl_scheme fedprox --mu 5 --alpha 0.5 --batch_size 64 --model resnet18 --hidden 512 --scheme All
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
```

#### For CIFAR100 with multiple local epochs with FedProx
```
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py --scheme RLU --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
```




#### For CIFAR10 with multiple local epochs with FedDyn

```
python main.py --local_epoch 5 --dataset CIFAR10 --momentum 0.0 --fl_scheme feddyn --weight_decay 1e-3 --alpha_coefficient 1e-2 --alpha 0.5 --batch_size 64 --model resnet18 --hidden 512
```

## VLI* theory and checkpoint diagnostics

`VLI_star_theory.tex` contains the derivations for the symmetric closed-form
estimator and the generalized estimator with supplied `(r, Q)`. The callable
implementations are `VLI_star_closed_form` and `VLI_star_generalized` in
`client/attacks.py`. Both take the accumulated **bias-gradient** vector, not
the weight-gradient feature sum used by the existing FedAvg `Client.VLI` path.
The generalized routine currently treats `r` and `Q` as known inputs; it does
not claim to infer them from a shared update.

Run the closed-form VLI* through the existing client evaluation loop with
FedAvg:

```bash
python main.py --scheme VLI_star --fl_scheme fedavg \
  --dataset CIFAR10 --model resnet18 --hidden 512 --n_classes 10 \
  --batch_size 64 --local_epochs 10 --alpha 0.5 \
  --check_point_path path/to/model.pth
```

This CLI path uses summed cross-entropy bias gradients and reports per-client
window-averaged cAcc and iAcc. `VLI_star_generalized` remains a callable
function because `main.py` has no principled source for oracle `r` and `Q`.
The CLI currently rejects FedProx and FedDyn for VLI*: their communicated
updates include additional objective/optimizer terms outside the closed-form
derivation.

Measure train/aux error, per-client difficulty ratio `kappa`, and class-wise
error-transition rows on checkpoints with:

```bash
python scripts/measure_vli_star_diagnostics.py \
  --checkpoint checkpoints/model_a.pth checkpoints/model_b.pth \
  --dataset CIFAR10 --model resnet18 --hidden 512 \
  --n-clients 10 --alpha 0.5 --client-ids all --prop 0.1 \
  --batch-size 64 --local-epochs 10
```

The script writes `summary.csv`, `Q_rows.csv`, and `window_recovery.csv` under
`results/vli_star_diagnostics/` by default. These are **oracle diagnostics**:
measuring `epsilon_train`, `r`, and train-derived rows of `Q` uses the private
client labels. Rows for absent or numerically zero-error classes are filled
from class-conditional aux data and explicitly marked; they do not affect
`Q^T a` when that class has zero error mass. `window_recovery.csv` compares the closed form and the
generalized estimator with oracle window-level `(r, Q)` against the true
window counts. This checks the identities and their assumptions on checkpoint
outputs, but does not show that `(r, Q)` are attacker-estimable.



#### Citeation
Please cite our paper, if you happen to use this code:
```
@inproceedings{
chen2024recovering,
title={Recovering Labels from Local Updates in Federated Learning},
author={Huancheng Chen and Haris Vikalo},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=E41gvBG4s6}
}
```
