# 🎬 SkyReels V2 - Google Colab Setup Guide

This guide helps you run SkyReels V2 with a beautiful web interface in Google Colab.

## 🚀 Quick Start

### Option 1: Use the Complete Colab Notebook
1. Open `SkyReels_V2_Colab.ipynb` in Google Colab
2. Ensure you have GPU enabled: `Runtime` → `Change runtime type` → `Hardware accelerator: GPU`
3. Run all cells in order
4. Use the web interface that appears at the bottom

### Option 2: Use the Simple Interface
1. Clone this repository in Colab
2. Run the simple interface:
```python
!python skyreels_gradio_interface.py
```

## 📋 Requirements

### Hugging Face Account
**IMPORTANT**: You need a Hugging Face account and access token:
1. Create account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with "Read" access
4. Copy the token for use in Colab

### GPU Requirements
- **Minimum**: T4 (15GB) - Use 1.3B models with CPU offload
- **Recommended**: A100 (40GB+) - Can run all models including 14B
- **Optimal**: A100 80GB - Run 14B models at 720P without offload

### Runtime Setup
```bash
# 1. Enable GPU in Colab
Runtime → Change runtime type → Hardware accelerator: GPU

# 2. Check your GPU
!nvidia-smi
```

## 🛠️ Installation Steps

### 1. Install Dependencies
```bash
# System dependencies
!apt-get update -qq
!apt-get install -y -qq ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0

# Python packages
!pip install -q torch==2.5.1 torchvision==0.20.1
!pip install -q diffusers>=0.31.0 transformers==4.49.0
!pip install -q accelerate==1.6.0 tqdm imageio easydict ftfy
!pip install -q opencv-python==4.10.0.84 imageio-ffmpeg
!pip install -q gradio spaces huggingface_hub
!pip install -q numpy>=1.23.5,<2

# Video processing dependencies
!pip install -q decord av
```

### 2. Clone Repository
```bash
!git clone https://github.com/SkyworkAI/SkyReels-V2.git
%cd SkyReels-V2
```

### 3. Launch Interface
```python
# Run the simple interface
!python skyreels_gradio_interface.py

# Or use the complete notebook interface
# (Run all cells in SkyReels_V2_Colab.ipynb)
```

## 🎨 Using the Interface

### Basic Usage
1. **Enter Prompt**: Describe the video you want to generate
2. **Choose Model**: 
   - 1.3B models: Faster, lower memory usage
   - 14B models: Higher quality, more memory needed
3. **Select Resolution**:
   - 540P: 544x960, up to 97 frames
   - 720P: 720x1280, up to 121 frames
4. **Click Generate**: Wait for the video to be created

### Advanced Settings
- **Guidance Scale**: Controls how closely the model follows the prompt (6.0 for T2V, 5.0 for I2V)
- **Shift**: Flow matching parameter (8.0 for T2V, 3.0 for I2V)
- **Inference Steps**: More steps = better quality but slower (50 recommended)
- **Seed**: Use -1 for random, or set a number for reproducible results
- **CPU Offload**: Reduces GPU memory usage but slower generation

### Image-to-Video
1. Upload an image in the "Input Image" section
2. Change "Generation Type" to "Image-to-Video"
3. Write a prompt describing the motion you want
4. The model will animate your image

## 📊 Model Recommendations

| GPU Memory | Recommended Model | Settings |
|------------|------------------|----------|
| 15GB (T4) | 1.3B-540P | CPU Offload: ON, Frames: 49-97 |
| 24GB (RTX 4090) | 14B-540P | CPU Offload: ON, Frames: 97 |
| 40GB (A100) | 14B-720P | CPU Offload: OFF, Frames: 121 |
| 80GB (A100) | All models | Max settings |

## 🎯 Example Prompts

### Nature
```
A majestic waterfall cascading down moss-covered rocks in a lush forest, 
with sunbeams filtering through the canopy and mist rising from the pool below.
```

### Urban
```
A bustling city street at night with neon lights reflecting on wet pavement, 
people walking with umbrellas, and cars passing by with light trails.
```

### Abstract
```
Colorful paint drops falling into water in slow motion, creating beautiful 
ripples and color mixing patterns, artistic and mesmerizing.
```

### Animals
```
A family of dolphins jumping gracefully out of crystal clear ocean water 
at sunset, with golden light reflecting on the waves.
```

## 🔧 Troubleshooting

### Common Issues

**"Out of Memory" Error**
```python
# Solutions:
# 1. Use smaller model
model_size = "1.3B"  # instead of "14B"

# 2. Enable CPU offload
use_offload = True

# 3. Reduce frames
num_frames = 49  # instead of 97

# 4. Use lower resolution
resolution = "540P"  # instead of "720P"

# 5. Clear memory
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

**"No GPU Available"**
- Go to `Runtime` → `Change runtime type`
- Set `Hardware accelerator` to `GPU`
- Restart runtime

**"Installation Failed" or "ModuleNotFoundError"**
```python
# Install missing dependencies
!pip install -q decord av dashscope

# Restart runtime and try again
!pip install --upgrade pip
!pip install --force-reinstall torch torchvision

# If specific module is missing:
!pip install -q decord  # For video processing
!pip install -q av      # For video encoding
```

**"Model Download Slow"**
```python
# Use alternative model source
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**"Access Token Required" or "Authentication Error"**
```python
# Set up Hugging Face token
from huggingface_hub import login
login(token="YOUR_TOKEN_HERE")

# Or set environment variable
import os
os.environ['HF_TOKEN'] = "YOUR_TOKEN_HERE"
```

### Performance Tips

1. **Faster Generation**:
   - Use 1.3B models
   - Reduce inference steps to 20-30
   - Enable CPU offload only if needed

2. **Better Quality**:
   - Use 14B models
   - Increase inference steps to 50-100
   - Use detailed prompts
   - Disable CPU offload if possible

3. **Memory Management**:
   - Generate one video at a time
   - Clear outputs after generation
   - Restart runtime if memory issues persist

## 📱 Sharing Your Interface

The interface automatically creates a public link when launched. You can:
1. Share the Gradio link with others
2. Keep it private by setting `share=False`
3. Access it from your phone or other devices

## 🎥 Output Management

Generated videos are saved in the `outputs/` folder with timestamps:
```
outputs/
├── video_20241201_143022_12345.mp4
├── video_20241201_143156_67890.mp4
└── ...
```

## 💡 Pro Tips

1. **Prompt Engineering**:
   - Use descriptive language with visual details
   - Specify lighting, camera angles, and movement
   - Mention colors, textures, and atmosphere
   - Keep prompts 20-100 words for best results

2. **Model Selection**:
   - Start with 1.3B models to test prompts
   - Use 14B models for final high-quality videos
   - I2V models work best with detailed prompts about motion

3. **Performance**:
   - Generate shorter videos first to test settings
   - Use consistent seeds for variations
   - Save successful prompts and settings

## 💽 Disk Space Management

Google Colab has limited disk space (~112GB). Large AI models can quickly fill this up.

### 🆘 Emergency Cleanup (When Disk is Full)
```python
# Run this immediately if disk is full:
!pip cache purge
!apt-get clean
!rm -rf /tmp/* /var/tmp/*
!rm -rf /root/.cache/huggingface/
!rm -rf /root/.cache/torch/
!rm -rf outputs/
!find /content -name '__pycache__' -type d -exec rm -rf {} +

# Check space
import shutil
total, used, free = shutil.disk_usage('/')
print(f"Free: {free//1024**3:.1f}GB")
```

### 💡 Space-Saving Tips
1. **Use smaller models** when possible (1.3B vs 14B)
2. **Generate shorter videos** (49-77 frames vs 97-121)
3. **Delete outputs** after downloading them
4. **Clear caches** regularly
5. **Restart runtime** to completely clear memory

### 📈 Colab Pro Benefits
- **200GB disk space** (vs 112GB free)
- **Faster GPUs** (A100, V100)
- **Longer runtime** (up to 24h vs 12h)
- **Priority access** during busy times

Consider upgrading if you frequently hit space limits.

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the emergency cleanup if disk is full
3. Restart the Colab runtime
4. Clear all outputs and run cells again
5. Check the original [SkyReels V2 repository](https://github.com/SkyworkAI/SkyReels-V2) for updates

---

Happy video generating! 🎬✨ 