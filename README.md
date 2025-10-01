# GenreDiscern V2 - Refactored Music Genre Classification System

## Current Version Stack
- Python 3.13.0
- PyTorch: 2.8.0+cu128 (compiled with CUDA 12.8)
- NVIDIA Driver Version: 535.247.01
- CUDA Toolkit: 13.0

## Setup (Debian)

- Use pyenv to install python 3.13.0
```
pyenv shell 3.13.0
python -m venv env
source env/bin/activate
```

- Install CUDA 13.0
```
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb
sudo dpkg -i cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb
sudo cp /var/cuda-repo-debian12-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

- Install dependencies
```
pip install -r requirements.txt
```

## Setup (Windows)

- Use pyenv to install python 3.13.0
```

```

- Install CUDA 13.0
```

```

- Install dependencies
```

```