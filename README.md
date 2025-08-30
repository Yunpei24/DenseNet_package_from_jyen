# DenseNet-PyTorch 🚀

A clean and modular PyTorch implementation of the research paper **"Densely Connected Convolutional Networks"** by Huang et al.

This project is structured as an installable Python package, making it ideal for research, experimentation, and integration into automated training pipelines (CI/CD).

-----

## DenseNet Architecture

The fundamental idea of DenseNet is to connect each layer to every other layer that follows it. For a layer $l$, the feature-maps of all preceding layers $x\_0, ..., x\_{l-1}$ are used as input:

$x\_l = H\_l([x\_0, x\_1, ..., x\_{l-1}])$

This encourages feature reuse, strengthens gradient propagation, and significantly reduces the number of parameters.

**(Add your architecture image here by replacing the path)**
\!(path/to/your/densenet\_architecture\_image.png)

-----

## Project Structure

The project is organized to separate the model logic (the library) from the executable scripts.

```bash
densenet-pytorch/
├── .github/workflows/      # Example workflows for Continuous Integration (CI/CD)
├── src/
│   └── densenet/           # The core Python package, installable via pip
│       ├── __init__.py
│       ├── model.py
│       ├── layers.py
│       └── ...
├── scripts/                # Scripts to run training
│   ├── run_cifar_training.py
│   └── run_imagenet_training.py
├── setup.py                # Configuration file to make the package installable
├── requirements.txt
└── README.md
```

-----

## Installation ⚙️

To use this project, first clone the repository and install the package in "editable" mode. This will allow you to modify the source code and use the changes instantly.

### 1\. Clone the Repository

```bash
git clone https://github.com/Yunpei24/DenseNet_package_from_jyen.git
cd DenseNet_package_from_jyen
```

### 2\. Install the Package

The `-e .` command installs the package locally while allowing you to modify it.

```bash
pip install -e .
```

-----

## Training Guide 🧠

The `scripts/` folder contains the entry points for running training sessions. Each script can be configured via command-line arguments.

### 1\. Training on CIFAR-10 / CIFAR-100

The `run_cifar_training.py` script is used for both CIFAR datasets. You can choose the dataset and configure the hyperparameters directly.

**Key Arguments:**

  * `--dataset`: The dataset to use (`cifar10` or `cifar100`). **Default: `cifar10`**.
  * `--growth-rate` or `-k`: The growth rate (k) of the model. **Default: `12`**.
  * `--epochs`: The total number of epochs. **Default: `300`**.
  * `--batch-size`: The batch size. **Default: `128`**.
  * `--lr`: The initial learning rate. **Default: `0.1`**.

#### **Example Commands:**

  * **Run a simple training on CIFAR-10:**

    ```bash
    python scripts/run_cifar_training.py --dataset cifar10 --growth-rate 12 --epochs 150
    ```

  * **Run a training on CIFAR-100 with a higher `growth-rate`:**

    ```bash
    python scripts/run_cifar_training.py --dataset cifar100 --growth-rate 24 --epochs 300 --batch-size 64
    ```

-----

### 2\. Training on ImageNet

Training on ImageNet is more complex and requires specific data paths. The `run_imagenet_training.py` script is designed to use the data structure provided by Kaggle.

**Key Arguments:**

  * `--input-dir`: **(Required)** The path to the root folder of the ImageNet dataset.
  * `--output-dir`: The folder where the sorted validation set will be created. **Default: `./`**.
  * `--model-arch`: The DenseNet architecture to use (`densenet121`, `densenet169`, etc.). **Default: `densenet121`**.
  * `--epochs`: The total number of epochs. **Default: `90`**.
  * `--batch-size`: The batch size (adjust according to your GPU's VRAM). **Default: `128`**.
  * `--num-workers`: The number of processes for data loading. **Default: `4`**.

#### **Example Command:**

  * **Run a training with DenseNet-121 on ImageNet:**
    ```bash
    python scripts/run_imagenet_training.py \
        --input-dir /path/to/your/imagenet-dataset \
        --output-dir ./imagenet_prepared \
        --model-arch densenet121 \
        --batch-size 256 \
        --num-workers 8
    ```
    > **Note:** The first run of this script will take some time as it needs to copy and organize the 50,000 images of the validation set.

-----

### 3\. Tracking Experiments with Weights & Biases (W\&B) 📊

Tracking with W\&B is enabled by default to log the metrics of your training runs.

  * **Configuration:** For this to work locally or on a server, make sure to set your W\&B API key as an environment variable:

    ```bash
    export WANDB_API_KEY="your_api_key_here"
    ```

    The script will detect it automatically. If you are on Kaggle, it will try to use Kaggle secrets.

  * **Disabling:** To run a training without W\&B, simply add the `--no-wandb` flag to any training command.

    ```bash
    python scripts/run_cifar_training.py --dataset cifar10 --no-wandb
    ```

-----

## Continuous Integration (CI/CD)

An example workflow for GitHub Actions can be found in `.github/workflows/training.yml`. It shows how to automate the installation of dependencies and the execution of a training script on each `push` to the `main` branch, which is ideal for validating that the code remains functional.

-----

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.