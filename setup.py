# Helper script if user wants to run without notebook
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
    print("Installing required specific versions of PyTorch...")
    run_cmd("pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124")
    
    print("Installing requirements from requirements.txt...")
    run_cmd("pip install -r requirements.txt")
    
    print("Pre-downloading VIZINTZOR/F5-TTS-TH-V2 model...")
    # ใช้ Python API เพื่อหลีกเลี่ยงปัญหาหน้าต่างแจ้งเตือน (Interactive Prompt [Y/n]) ที่ทำให้ Colab ค้าง
    run_cmd("python -c \"from huggingface_hub import snapshot_download; snapshot_download('VIZINTZOR/F5-TTS-TH-V2')\"")
    
    print("Setup complete! You can now run `python app.py`.")
