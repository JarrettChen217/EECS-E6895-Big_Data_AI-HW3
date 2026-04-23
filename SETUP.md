# Environment Setup (Ubuntu + RTX 3070)

## 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install PyTorch with CUDA

Check your CUDA version first:
```bash
nvcc --version
```

Then install the matching PyTorch build:

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> If CUDA 13 wheels are available at https://download.pytorch.org/whl/, use `cu130` instead.

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Install all other dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Jupyter server (accessible from your local machine)

```bash
jupyter lab \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888
```

Copy the token URL printed in the terminal, e.g.:
```
http://127.0.0.1:8888/lab?token=abc123...
```

On your **local machine**, replace `127.0.0.1` with the Ubuntu machine's IP address:
```
http://<ubuntu-ip>:8888/lab?token=abc123...
```

> Make sure port 8888 is open on the Ubuntu machine's firewall:
> ```bash
> sudo ufw allow 8888
> ```

## 5. (Optional) Set a persistent password instead of a token

```bash
jupyter lab password
# Enter your chosen password when prompted
```

Then start without token:
```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```
