#!/usr/bin/env python3
"""
SkyReels V2 Gradio Interface
A user-friendly web interface for generating videos with SkyReels V2 models.

Usage:
    python skyreels_v2_gradio_app.py [--port PORT] [--share]

Options:
    --port PORT     Port to run the interface on (default: 7860)
    --share         Create a public share link
    --help          Show this help message
"""

import argparse
import gc
import os
import random
import time
import traceback
from typing import Optional, Tuple

import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image
from diffusers.utils import load_image

# Import SkyReels modules
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline, Text2VideoPipeline
from skyreels_v2_infer.pipelines import PromptEnhancer, resizecrop


class SkyReelsV2Interface:
    """Main interface class for SkyReels V2 video generation."""
    
    def __init__(self):
        """Initialize the interface."""
        self.current_pipeline = None
        self.current_model_id = None
        self._check_gpu()
        
    def _check_gpu(self):
        """Check GPU availability and memory."""
        if not torch.cuda.is_available():
            print("❌ Warning: No GPU detected. This may cause issues or very slow performance.")
            return
            
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 12:
            print("⚠️  Warning: Low GPU memory. Recommend using 1.3B models with CPU offload.")
        elif gpu_memory < 20:
            print("📝 Recommended: 1.3B models or 14B with heavy offloading.")
        else:
            print("📝 Recommended: All models supported.")
    
    def generate_video(self, 
                      prompt: str, 
                      image: Optional[np.ndarray],
                      model_type: str,
                      model_size: str,
                      resolution: str,
                      num_frames: int,
                      guidance_scale: float,
                      shift: float,
                      inference_steps: int,
                      fps: int,
                      seed: int,
                      use_prompt_enhancer: bool,
                      use_offload: bool,
                      progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """
        Generate a video based on the provided parameters.
        
        Args:
            prompt: Text description for the video
            image: Optional input image for I2V
            model_type: "Text-to-Video" or "Image-to-Video"
            model_size: "1.3B" or "14B"
            resolution: "540P" or "720P" 
            num_frames: Number of frames to generate
            guidance_scale: Guidance scale for generation
            shift: Shift parameter for flow matching
            inference_steps: Number of denoising steps
            fps: Frames per second for output video
            seed: Random seed (-1 for random)
            use_prompt_enhancer: Whether to enhance the prompt
            use_offload: Whether to use CPU offloading
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (video_path, info_message)
        """
        try:
            progress(0, desc="Initializing...")
            
            # Validate inputs
            if not prompt.strip():
                return None, "❌ Error: Please provide a prompt."
            
            if model_type == "Image-to-Video" and image is None:
                return None, "❌ Error: Please upload an image for Image-to-Video generation."
            
            # Determine model ID
            model_prefix = "T2V" if model_type == "Text-to-Video" else "I2V"
            if model_size == "14B":
                model_id = f"Skywork/SkyReels-V2-{model_prefix}-14B-{resolution}"
            else:
                # Handle 1.3B models (only some variants available)
                if model_prefix == "T2V" and resolution == "720P":
                    return None, "❌ Error: 1.3B T2V model not available in 720P. Use 540P or 14B model."
                model_id = f"Skywork/SkyReels-V2-{model_prefix}-1.3B-{resolution}"
            
            progress(0.1, desc="Downloading model...")
            try:
                model_path = download_model(model_id)
            except Exception as e:
                return None, f"❌ Error downloading model: {str(e)}"
            
            # Set seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            # Set dimensions
            if resolution == "540P":
                height, width = 544, 960
            else:  # 720P
                height, width = 720, 1280
            
            progress(0.2, desc="Processing prompt...")
            prompt_input = prompt.strip()
            
            # Enhance prompt if requested and no image provided
            if use_prompt_enhancer and image is None and model_type == "Text-to-Video":
                try:
                    progress(0.25, desc="Enhancing prompt...")
                    enhancer = PromptEnhancer()
                    enhanced_prompt = enhancer(prompt_input)
                    if enhanced_prompt and enhanced_prompt.strip():
                        prompt_input = enhanced_prompt.strip()
                        print(f"Enhanced prompt: {prompt_input}")
                    del enhancer
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Prompt enhancement failed: {e}")
                    # Continue with original prompt
            
            progress(0.3, desc="Loading pipeline...")
            
            # Initialize or reuse pipeline
            if self.current_model_id != model_id or self.current_pipeline is None:
                # Clean up previous pipeline
                if self.current_pipeline:
                    del self.current_pipeline
                    torch.cuda.empty_cache()
                
                # Create new pipeline
                if model_type == "Text-to-Video":
                    self.current_pipeline = Text2VideoPipeline(
                        model_path=model_path, 
                        dit_path=model_path, 
                        use_usp=False, 
                        offload=use_offload
                    )
                else:  # Image-to-Video
                    self.current_pipeline = Image2VideoPipeline(
                        model_path=model_path, 
                        dit_path=model_path, 
                        use_usp=False, 
                        offload=use_offload
                    )
                
                self.current_model_id = model_id
            
            # Process input image if provided
            input_image = None
            if image is not None:
                progress(0.4, desc="Processing input image...")
                input_image = Image.fromarray(image).convert("RGB")
                image_width, image_height = input_image.size
                
                # Adjust dimensions for portrait images
                if image_height > image_width:
                    height, width = width, height
                
                input_image = resizecrop(input_image, height, width)
            
            progress(0.5, desc="Generating video...")
            
            # Prepare generation parameters
            negative_prompt = ("Bright tones, overexposed, static, blurred details, subtitles, "
                             "style, works, paintings, images, static, overall gray, worst quality, "
                             "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                             "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                             "misshapen limbs, fused fingers, still picture, messy background, "
                             "three legs, many people in the background, walking backwards")
            
            kwargs = {
                "prompt": prompt_input,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "generator": torch.Generator(device="cuda").manual_seed(seed),
                "height": height,
                "width": width,
            }
            
            if input_image is not None:
                kwargs["image"] = input_image
            
            # Generate video
            print(f"Starting generation with parameters: {kwargs}")
            with torch.cuda.amp.autocast(dtype=self.current_pipeline.transformer.dtype), torch.no_grad():
                video_frames = self.current_pipeline(**kwargs)[0]
            
            progress(0.9, desc="Saving video...")
            
            # Save video
            os.makedirs("outputs", exist_ok=True)
            current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            video_filename = f"skyreels_v2_{current_time}_{seed}.mp4"
            video_path = os.path.join("outputs", video_filename)
            
            imageio.mimwrite(
                video_path, 
                video_frames, 
                fps=fps, 
                quality=8, 
                output_params=["-loglevel", "error"]
            )
            
            progress(1.0, desc="Complete!")
            
            # Generate info message
            info_msg = (f"✅ Video generated successfully!\n"
                       f"Model: {model_id}\n" 
                       f"Seed: {seed}\n"
                       f"Frames: {len(video_frames)}\n"
                       f"Resolution: {width}x{height}\n"
                       f"Duration: {len(video_frames)/fps:.1f}s\n"
                       f"Output: {video_path}")
            
            return video_path, info_msg
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"Full error: {traceback.format_exc()}")
            # Clean up on error
            torch.cuda.empty_cache()
            gc.collect()
            return None, error_msg


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    # Initialize interface
    skyreels_interface = SkyReelsV2Interface()
    
    # Create Gradio app
    with gr.Blocks(
        title="SkyReels V2 - Video Generator", 
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center; 
            margin-bottom: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎬 SkyReels V2 - Infinite-Length Film Generator</h1>
            <p>Generate high-quality videos with state-of-the-art AI models</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Input controls
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Configuration")
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3,
                    value="A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface."
                )
                
                image = gr.Image(
                    label="Input Image (Optional - for Image-to-Video)",
                    type="numpy"
                )
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=["Text-to-Video", "Image-to-Video"],
                        value="Text-to-Video",
                        label="Generation Type"
                    )
                    
                    model_size = gr.Dropdown(
                        choices=["1.3B", "14B"],
                        value="1.3B",
                        label="Model Size"
                    )
                
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=["540P", "720P"],
                        value="540P",
                        label="Resolution"
                    )
                    
                    num_frames = gr.Slider(
                        minimum=49,
                        maximum=121,
                        value=97,
                        step=1,
                        label="Number of Frames"
                    )
                
                gr.Markdown("### ⚙️ Advanced Settings")
                
                with gr.Row():
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=6.0,
                        step=0.5,
                        label="Guidance Scale"
                    )
                    
                    shift = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=8.0,
                        step=0.5,
                        label="Shift"
                    )
                
                with gr.Row():
                    inference_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Inference Steps"
                    )
                    
                    fps = gr.Slider(
                        minimum=8,
                        maximum=30,
                        value=24,
                        step=1,
                        label="FPS"
                    )
                
                seed = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0
                )
                
                with gr.Row():
                    use_prompt_enhancer = gr.Checkbox(
                        label="Enhance Prompt",
                        value=False
                    )
                    
                    use_offload = gr.Checkbox(
                        label="CPU Offload (reduces VRAM)",
                        value=True
                    )
                
                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output and info
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Generated Video")
                
                output_video = gr.Video(
                    label="Generated Video",
                    height=400
                )
                
                output_info = gr.Textbox(
                    label="Generation Info",
                    lines=6,
                    interactive=False
                )
                
                gr.Markdown("""
                ### 💡 Tips:
                - **1.3B models**: Faster, lower VRAM usage (~15GB)
                - **14B models**: Higher quality, more VRAM (~45GB)  
                - **540P**: 544x960 resolution, 97 frames max
                - **720P**: 720x1280 resolution, 121 frames max
                - **CPU Offload**: Reduces VRAM usage but slower
                - **Image-to-Video**: Upload an image for I2V generation
                - **Prompt Enhancement**: Uses AI to improve text prompts
                """)
        
        # Auto-adjust parameters based on model type
        def update_params(model_type_val):
            if model_type_val == "Image-to-Video":
                return gr.update(value=5.0), gr.update(value=3.0)  # guidance_scale, shift
            else:
                return gr.update(value=6.0), gr.update(value=8.0)
        
        model_type.change(
            update_params,
            inputs=[model_type],
            outputs=[guidance_scale, shift]
        )
        
        # Auto-adjust frames based on resolution
        def update_frames(resolution_val):
            if resolution_val == "720P":
                return gr.update(maximum=121, value=121)
            else:
                return gr.update(maximum=97, value=97)
        
        resolution.change(
            update_frames,
            inputs=[resolution],
            outputs=[num_frames]
        )
        
        # Generate video on button click
        generate_btn.click(
            skyreels_interface.generate_video,
            inputs=[
                prompt, image, model_type, model_size, resolution,
                num_frames, guidance_scale, shift, inference_steps,
                fps, seed, use_prompt_enhancer, use_offload
            ],
            outputs=[output_video, output_info]
        )
        
        # Add examples
        gr.Markdown("### 📚 Example Prompts")
        
        example_prompts = [
            ["A majestic waterfall cascading down moss-covered rocks in a lush forest, with sunbeams filtering through the canopy and mist rising from the pool below.", None, "Text-to-Video"],
            ["A bustling city street at night with neon lights reflecting on wet pavement, people walking with umbrellas, and cars passing by with light trails.", None, "Text-to-Video"],
            ["Colorful paint drops falling into water in slow motion, creating beautiful ripples and color mixing patterns, artistic and mesmerizing.", None, "Text-to-Video"],
            ["A family of dolphins jumping gracefully out of crystal clear ocean water at sunset, with golden light reflecting on the waves.", None, "Text-to-Video"]
        ]
        
        gr.Examples(
            examples=example_prompts,
            inputs=[prompt, image, model_type],
            label="Click to load example prompts"
        )
    
    return demo


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7860, help="Port to run the interface on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server name/IP to bind to")
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_interface()
    
    print(f"🚀 Starting SkyReels V2 Interface on port {args.port}")
    print(f"📱 Share link: {'Enabled' if args.share else 'Disabled'}")
    
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
        show_error=True,
        debug=False
    )


if __name__ == "__main__":
    main() 