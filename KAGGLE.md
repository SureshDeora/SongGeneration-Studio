# Running on Kaggle

This repository includes support for running on Kaggle kernels.

## Instructions

1.  **Clone the Repository**:
    In a new Kaggle Notebook, clone this repository:
    ```bash
    !git clone https://github.com/BazedFrog/SongGeneration-Studio
    %cd SongGeneration-Studio
    ```

2.  **Open the Notebook**:
    You can simply open and run the provided `kaggle_notebook.ipynb` file if you have uploaded it, or copy the content of the cells below into your notebook.

## Quick Start (Copy & Paste)

If you are starting a fresh notebook on Kaggle, follow these steps:

### Cell 1: Setup & Install
```python
# Clone the repository
!git clone https://github.com/BazedFrog/SongGeneration-Studio
%cd SongGeneration-Studio

# Install System Dependencies
!apt-get update && apt-get install -y ffmpeg

# Install Python Dependencies
!pip install -r requirements.txt
!pip install -r requirements_nodeps.txt --no-deps
!pip install pyngrok

# Run Setup (Downloads models ~15GB, takes a few minutes)
!python setup_kaggle.py
```

### Cell 2: Run Server
```python
import getpass
from pyngrok import ngrok, conf

# Authenticate ngrok
print("Enter your ngrok authtoken:")
token = getpass.getpass()
conf.get_default().auth_token = token

# Kill existing tunnels
ngrok.kill()

# Open tunnel
public_url = ngrok.connect(8000).public_url
print(f"\nTunnel Open! Access the Studio here: {public_url}\n")

# Run App
%cd app
!python main.py --port 8000
```
