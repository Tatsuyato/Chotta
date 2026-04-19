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
    # เปิดโหมด hf_transfer เพื่อเร่งความเร็ว x10 และข้ามไฟล์ Optimizer/Checkpoint เก่าที่ไม่ได้ใช้ (ลดขนาดจาก 35GB ให้เหลือเท่าที่จำเป็น)
    run_cmd("python -c \"import os; os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'; from huggingface_hub import snapshot_download; snapshot_download('VIZINTZOR/F5-TTS-TH-V2', ignore_patterns=['*optimizer*', '*optim*', '*checkpoint*'])\"")
    
    print("Setup complete! You can now run `python app.py`.")
