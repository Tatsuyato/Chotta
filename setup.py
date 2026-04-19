# Helper script if user wants to run without notebook
import subprocess
import sys
import importlib.util

def is_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
    print("1. Checking PyTorch installation...")
    try:
        import torch
        if not torch.__version__.startswith("2.4.0"):
            print(f"Found PyTorch {torch.__version__}, but require 2.4.0. Reinstalling...")
            run_cmd(f"{sys.executable} -m pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124")
        else:
            print("✅ PyTorch 2.4.0 is already installed. Skipping.")
    except ImportError:
        print("PyTorch not found. Installing specific versions...")
        run_cmd(f"{sys.executable} -m pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124")
        
    print("2. Checking requirements...")
    main_packages = ["gradio", "faster_whisper", "f5_tts_th", "mediapipe", "demucs", "yt_dlp", "hf_transfer"]
    missing = [pkg for pkg in main_packages if not is_installed(pkg)]
    
    if missing:
        print(f"Missing packages: {missing}. Installing from requirements.txt...")
        run_cmd(f"{sys.executable} -m pip install -q -r requirements.txt")
    else:
        print("✅ All main requirements are already installed. Skipping.")
    
    print("3. Checking VIZINTZOR/F5-TTS-TH-V2 model...")
    run_cmd(f"{sys.executable} -c \"import os; os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'; from huggingface_hub import snapshot_download; snapshot_download('VIZINTZOR/F5-TTS-TH-V2', ignore_patterns=['*optimizer*', '*optim*', '*checkpoint*'])\"")
    
    print("Setup complete! You can now run `python app.py`.")
