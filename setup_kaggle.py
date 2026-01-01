import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(command, check=True):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=check)

def setup():
    base_dir = Path.cwd()
    app_dir = base_dir / "app"
    
    # 1. Clone the SongGeneration repo
    if not app_dir.exists():
        print("Cloning SongGeneration...")
        run_command("git clone https://github.com/tencent-ailab/SongGeneration app")
    else:
        print("SongGeneration repo already exists.")

    # 2. Download model weights
    print("Downloading models... (this may take a while)")
    try:
        from huggingface_hub import snapshot_download
        
        # Download runtime files (ckpt, third_party)
        print("Downloading lglg666/SongGeneration-Runtime to app/...")
        snapshot_download(repo_id="lglg666/SongGeneration-Runtime", local_dir="app", local_dir_use_symlinks=False)
        
        # Download the base_full model (default recommendation for 12GB+ VRAM, fits in T4)
        # We download it to app/songgeneration_base_full as expected by models.py
        model_name = "songgeneration_base_full"
        model_repo = "lglg666/SongGeneration-base-full"
        model_dir = app_dir / model_name
        
        print(f"Downloading {model_repo} to {model_dir}...")
        snapshot_download(repo_id=model_repo, local_dir=str(model_dir), local_dir_use_symlinks=False)
        
    except ImportError:
        print("huggingface_hub not found. Please install it.")
        return
    except Exception as e:
        print(f"Error downloading models: {e}")
        return

    # 3. Copy files
    files_to_copy = [
        ("requirements.txt", "app/requirements.txt"),
        ("requirements_nodeps.txt", "app/requirements_nodeps.txt"),
        ("main.py", "app/main.py"),
        ("generation.py", "app/generation.py"),
        ("model_server.py", "app/model_server.py"),
        ("models.py", "app/models.py"),
        ("gpu.py", "app/gpu.py"),
        ("config.py", "app/config.py"),
        ("schemas.py", "app/schemas.py"),
        ("sse.py", "app/sse.py"),
        ("timing.py", "app/timing.py"),
        
        ("web/static/index.html", "app/web/static/index.html"),
        ("web/static/styles.css", "app/web/static/styles.css"),
        ("web/static/app.js", "app/web/static/app.js"),
        ("web/static/components.js", "app/web/static/components.js"),
        ("web/static/hooks.js", "app/web/static/hooks.js"),
        ("web/static/api.js", "app/web/static/api.js"),
        ("web/static/constants.js", "app/web/static/constants.js"),
        ("web/static/icons.js", "app/web/static/icons.js"),
        ("web/static/Logo_1.png", "app/web/static/Logo_1.png"),
        ("web/static/default.jpg", "app/web/static/default.jpg"),
        
        ("patches/builders.py", "app/codeclm/models/builders.py"),
        ("patches/demucs/apply.py", "app/third_party/demucs/models/apply.py"),
        ("patches/gradio/levo_inference_lowmem.py", "app/tools/gradio/levo_inference_lowmem.py"),
        ("patches/utils.py", "app/codeclm/utils/utils.py"),
    ]

    for src, dest in files_to_copy:
        src_path = base_dir / src
        dest_path = base_dir / dest
        
        if not src_path.exists():
            print(f"Warning: Source file {src} does not exist!")
            continue
            
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src} -> {dest}")

    print("Setup complete.")

if __name__ == "__main__":
    setup()
