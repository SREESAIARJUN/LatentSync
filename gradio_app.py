import gradio as gr
from pathlib import Path
from datetime import datetime
import argparse


CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")



def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
):
    import modal
    
    # Connect to Modal - use Function.from_name for deployed functions (Modal SDK 1.0+)
    # For class methods, use "ClassName.method_name" format
    infer_fn = modal.Function.from_name("latentsync", "LatentSync.infer")
    
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(video_path).stem
    output_path = output_dir / f"{video_stem}_{current_time}.mp4"

    print(f"Processing video: {video_path} with audio: {audio_path}")

    # Read inputs
    with open(video_path, "rb") as v:
        video_bytes = v.read()
    with open(audio_path, "rb") as a:
        audio_bytes = a.read()
        
    try:
        print("Calling Modal LatentSync.infer...")
        # Call remote function
        out_bytes = infer_fn.remote(
            video_bytes=video_bytes, 
            audio_bytes=audio_bytes, 
            seed=int(seed), 
            guidance_scale=float(guidance_scale), 
            inference_steps=int(inference_steps)
        )
        
        # Write output
        with open(output_path, "wb") as o:
            o.write(out_bytes)
            
        print("Processing completed successfully.")
        return str(output_path)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")



# Create Gradio interface
with gr.Blocks(title="LatentSync demo") as demo:
    gr.Markdown(
        """
    <h1 align="center">LatentSync</h1>

    <div style="display:flex;justify-content:center;column-gap:4px;">
        <a href="https://github.com/bytedance/LatentSync">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a> 
        <a href="https://arxiv.org/abs/2412.09262">
            <img src='https://img.shields.io/badge/arXiv-Paper-red'>
        </a>
    </div>
    """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video")
            audio_input = gr.Audio(label="Input Audio", type="filepath")

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=1.5,
                    step=0.1,
                    label="Guidance Scale",
                )
                inference_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps")

            with gr.Row():
                seed = gr.Number(value=1247, label="Random Seed", precision=0)

            process_btn = gr.Button("Process Video")

        with gr.Column():
            video_output = gr.Video(label="Output Video")

            gr.Examples(
                examples=[
                    ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                    ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                    ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                ],
                inputs=[video_input, audio_input],
            )

    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
            seed,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)
