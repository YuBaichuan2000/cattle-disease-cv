# Environment

## System Requirements

A GPU with 16GB VMem Above, we suggest 3090/A5000/V100/A100 GPU for training, and T4 for inference, we utilized a single 4090 for development and performed training the A100-40G cluster.

There are the GPU options on Google Cloud : https://cloud.google.com/compute/docs/gpus

For more cost effective options you can explore : https://vast.ai/ 

- Ubuntu 20.04 / 22.04 : https://cloud.google.com/deep-learning-vm/docs/introduction
- Anaconda/Miniconda : https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
- Python 3.8^
- PyTorch 1.13^ : https://pytorch.org/get-started/previous-versions/#v1131
- Torchlightning

You can then share or move this [environment.yml](../environment.yml) file to another machine or location. To recreate the environment on another system, one would use:
```bash
conda env create -f environment.yml
```
