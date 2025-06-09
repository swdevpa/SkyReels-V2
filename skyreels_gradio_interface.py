#!/usr/bin/env python3
"""
SkyReels V2 Gradio Interface
Simple web interface for video generation
"""

import gradio as gr
import torch
import os
import time
import random
import imageio
import gc
from PIL import Image

# Import SkyReels modules
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline, Text2VideoPipeline
from skyreels_v2_infer.pipelines import resizecrop


class VideoGenerator:
    def __init__(self):
        self.pipeline = None
        self.current_model = None
    
    def generate_video(self, prompt, image, model_type, model_size, resolution, 
                      num_frames, guidance_scale, shift, steps, fps, seed, 
                      use_offload, progress=gr.Progress()):
        
        try:
            progress(0, "Starting...")
            
            # Validate model availability
            available_combos = {
                ("Text-to-Video", "14B", "540P"): "Skywork/SkyReels-V2-T2V-14B-540P",
                ("Text-to-Video", "14B", "720P"): "Skywork/SkyReels-V2-T2V-14B-720P",
                ("Image-to-Video", "1.3B", "540P"): "Skywork/SkyReels-V2-I2V-1.3B-540P",
                ("Image-to-Video", "14B", "540P"): "Skywork/SkyReels-V2-I2V-14B-540P",
                ("Image-to-Video", "14B", "720P"): "Skywork/SkyReels-V2-I2V-14B-720P",
            }
            
            combo = (model_type, model_size, resolution)
            if combo not in available_combos:
                error_msg = f"❌ Model combination not available: {model_type} {model_size} {resolution}\n"
                error_msg += "✅ Available combinations:\n"
                for (mt, ms, res), model_id in available_combos.items():
                    error_msg += f"   - {mt} {ms} {res}\n"
                return None, error_msg
            
            model_id = available_combos[combo]
            
            progress(0.1, "Loading model...")
            model_path = download_model(model_id)
            
            # Set random seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            # Set dimensions
            height, width = (544, 960) if resolution == "540P" else (720, 1280)
            
            progress(0.3, "Initializing pipeline...")
            
            # Create pipeline if needed
            if self.current_model != model_id:
                if self.pipeline:
                    del self.pipeline
                    torch.cuda.empty_cache()
                
                if model_type == "Text-to-Video":
                    self.pipeline = Text2VideoPipeline(
                        model_path=model_path, 
                        dit_path=model_path, 
                        use_usp=False, 
                        offload=use_offload
                    )
                else:
                    self.pipeline = Image2VideoPipeline(
                        model_path=model_path, 
                        dit_path=model_path, 
                        use_usp=False, 
                        offload=use_offload
                    )
                
                self.current_model = model_id
            
            progress(0.5, "Generating video...")
            
            # Prepare inputs
            kwargs = {
                "prompt": prompt,
                "negative_prompt": "worst quality, low quality, static",
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "generator": torch.Generator(device="cuda").manual_seed(seed),
                "height": height,
                "width": width,
            }
            
            # Add image if provided
            if image is not None:
                img = Image.fromarray(image).convert("RGB")
                if img.height > img.width:
                    height, width = width, height
                kwargs["image"] = resizecrop(img, height, width)
            
            # Generate
            with torch.cuda.amp.autocast(dtype=self.pipeline.transformer.dtype), torch.no_grad():
                frames = self.pipeline(**kwargs)[0]
            
            progress(0.9, "Saving...")
            
            # Save video
            os.makedirs("outputs", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}_{seed}.mp4"
            path = os.path.join("outputs", filename)
            
            imageio.mimwrite(path, frames, fps=fps, quality=8)
            
            # Clear GPU memory after generation
            del frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            progress(1.0, "Done!")
            
            info = f"Generated video at {width}x{height}\nSeed: {seed}\nFile: {filename}"
            return path, info
            
        except Exception as e:
            return None, f"Error: {str(e)}"


# Create generator
generator = VideoGenerator()

# Create interface
def create_ui():
    with gr.Blocks(title="SkyReels V2", theme=gr.themes.Soft()) as app:
        gr.HTML("<h1 align='center'>🎬 SkyReels V2 Video Generator</h1>")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt", 
                    lines=3,
                    value="A beautiful swan swimming in a serene lake"
                )
                
                image = gr.Image(label="Input Image (optional)", type="numpy")
                
                with gr.Row():
                    model_type = gr.Dropdown(
                        ["Text-to-Video", "Image-to-Video"],
                        value="Text-to-Video",
                        label="Type"
                    )
                    model_size = gr.Dropdown(["1.3B", "14B"], value="1.3B", label="Size")
                
                with gr.Row():
                    resolution = gr.Dropdown(["540P", "720P"], value="540P", label="Resolution")
                    num_frames = gr.Slider(49, 121, 97, label="Frames")
                
                with gr.Row():
                    guidance_scale = gr.Slider(1, 10, 6, label="Guidance")
                    shift = gr.Slider(1, 10, 8, label="Shift")
                
                with gr.Row():
                    steps = gr.Slider(10, 100, 50, label="Steps")
                    fps = gr.Slider(8, 30, 24, label="FPS")
                
                seed = gr.Number(label="Seed (-1 = random)", value=-1)
                use_offload = gr.Checkbox(label="CPU Offload", value=True)
                
                btn = gr.Button("Generate Video", variant="primary")
            
            with gr.Column():
                video_out = gr.Video(label="Generated Video")
                info_out = gr.Textbox(label="Info", lines=3)
        
        btn.click(
            generator.generate_video,
            inputs=[prompt, image, model_type, model_size, resolution, 
                   num_frames, guidance_scale, shift, steps, fps, seed, use_offload],
            outputs=[video_out, info_out]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True) 