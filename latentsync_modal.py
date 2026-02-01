
import modal
import os
import subprocess
import time
from pathlib import Path

# Define the image with necessary dependencies
# We use a CUDA-enabled base image implicitly by installing torch with CUDA support
# or relying on Modal's handling of GPU requests.
# However, requirements.txt specifies torch 2.5.1 + cu121.
# Let's ensure system dependencies are present.

def download_weights_script():
    # This script will be run at build time or runtime to populate the volume
    pass

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "ffmpeg", "wget", "tar", "libgl1", "libglib2.0-0", 
        "libsm6", "libxext6", "libxrender-dev", "cmake", "build-essential", 
        "python3-dev", "libopenblas-dev", "liblapack-dev",
        # CUDA runtime libraries for onnxruntime-gpu
        "gnupg2", "ca-certificates"
    )
    .run_commands(
        # Add NVIDIA CUDA repository and install CUDA runtime
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-1 libgl1 libglib2.0-0 ffmpeg"
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH", "PATH": "/usr/local/cuda-12.1/bin:$PATH"})
    .run_commands("python -m pip install --upgrade pip")
    .run_commands("pip install cython setuptools wheel numpy==1.26.4")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("onnxruntime-gpu==1.17.1")
    .pip_install_from_requirements("requirements.txt")
    # Install insightface using pip_install with build isolation
    .pip_install("insightface==0.7.3")
    # Install huggingface_hub for downloading weights
    .pip_install("huggingface_hub")
    .add_local_dir("latentsync", remote_path="/root/latentsync")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

app = modal.App("latentsync")

# Volume to store the large model weights
model_volume = modal.Volume.from_name("latentsync-weights", create_if_missing=True)
CHECKPOINTS_DIR = Path("/checkpoints")

# HuggingFace Hub repo for weights - same as HF Space
HF_REPO_ID = "ByteDance/LatentSync"

@app.function(image=image, volumes={CHECKPOINTS_DIR: model_volume}, timeout=3600)
def download_models():
    from huggingface_hub import snapshot_download
    
    # Check if weights already exist
    unet_ckpt = CHECKPOINTS_DIR / "latentsync_unet.pt"
    if unet_ckpt.exists():
        print("Weights appear to be present.")
        return

    print(f"Downloading weights from HuggingFace Hub: {HF_REPO_ID}...")
    
    # Download from HuggingFace Hub like the HF Space does
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(CHECKPOINTS_DIR)
    )
    
    print("Weights downloaded from HuggingFace Hub.")

    # Symlink logic from predict.py
    # os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
    # os.system("ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
    # We might need to do this at runtime in the class setup since ~ is ephemeral.

@app.cls(gpu="L40S", image=image, volumes={CHECKPOINTS_DIR: model_volume}, timeout=1800)
class LatentSync:
    @modal.enter()
    def setup(self):
        import torch
        from omegaconf import OmegaConf
        from diffusers import AutoencoderKL, DDIMScheduler
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature
        
        # Create symlink from volume (/checkpoints) to ./checkpoints to match HuggingFace paths
        local_checkpoints = Path("checkpoints")
        if not local_checkpoints.exists():
            os.symlink(str(CHECKPOINTS_DIR), str(local_checkpoints))
        
        # Use exact same paths as HuggingFace Space
        inference_ckpt_path = "checkpoints/latentsync_unet.pt"
        unet_config_path = "configs/unet/second_stage.yaml"
        
        if not Path(inference_ckpt_path).exists():
            raise FileNotFoundError(f"LatentSync weights not found. Please run 'modal run latentsync_modal.py::download_models' first.")
        
        self.config = OmegaConf.load(unet_config_path)
        
        # VGG16 symlink for torch hub cache
        cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        vgg_src = Path("checkpoints/auxiliary/vgg16-397923af.pth")
        vgg_dest = cache_dir / "vgg16-397923af.pth"
        if vgg_src.exists() and not vgg_dest.exists():
            os.symlink(vgg_src.absolute(), vgg_dest)
        
        # Check FP16 support
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        self.dtype = torch.float16 if is_fp16_supported else torch.float32
        
        scheduler = DDIMScheduler.from_pretrained("configs")
        
        # Whisper model path - exact same logic as HF Space
        if self.config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif self.config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
        
        self.audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=self.config.data.num_frames,
        )
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        self.vae.config.scaling_factor = 0.18215
        self.vae.config.shift_factor = 0
        
        self.unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            inference_ckpt_path,
            device="cpu",
        )
        self.unet = self.unet.to(dtype=self.dtype)
        
        self.pipeline = LipsyncPipeline(
            vae=self.vae,
            audio_encoder=self.audio_encoder,
            unet=self.unet,
            scheduler=scheduler,
        ).to("cuda")

    @modal.method()
    def infer(self, video_bytes: bytes, audio_bytes: bytes, seed: int = 1247, guidance_scale: float = 1.5, inference_steps: int = 20):
        import torch
        from accelerate.utils import set_seed
        import tempfile
        import subprocess
        
        print("="*60)
        print("LATENTSYNC INFERENCE DEBUG LOG")
        print("="*60)
        
        # Save inputs to temporary files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_file:
            vid_file.write(video_bytes)
            video_path = vid_file.name
            print(f"[DEBUG] Video saved to: {video_path}")
            print(f"[DEBUG] Video file size: {len(video_bytes)} bytes")
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as aud_file:
            aud_file.write(audio_bytes)
            audio_path = aud_file.name
            print(f"[DEBUG] Audio saved to: {audio_path}")
            print(f"[DEBUG] Audio file size: {len(audio_bytes)} bytes")
        
        # Check audio file with ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", audio_path],
                capture_output=True, text=True
            )
            print(f"[DEBUG] Audio ffprobe output: {result.stdout[:500] if result.stdout else 'No output'}")
        except Exception as e:
            print(f"[DEBUG] ffprobe error: {e}")
            
        output_path = "/tmp/output.mp4"
        
        if seed != -1:
            set_seed(seed)
            print(f"[DEBUG] Set seed to: {seed}")
        else:
            torch.seed()
            print(f"[DEBUG] Using random seed: {torch.initial_seed()}")
        
        print(f"[DEBUG] Config num_frames: {self.config.data.num_frames}")
        print(f"[DEBUG] Config resolution: {self.config.data.resolution}")
        print(f"[DEBUG] Config inference_steps: {self.config.run.inference_steps}")
        print(f"[DEBUG] Using guidance_scale: 1.0")
        print(f"[DEBUG] Using weight_dtype: {self.dtype}")
        
        # Test whisper audio encoder directly
        print("\n[DEBUG] Testing Audio Encoder (Whisper)...")
        try:
            import librosa
            import numpy as np
            
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            print(f"[DEBUG] Audio loaded: duration={len(audio_data)/sr:.2f}s, sr={sr}, shape={audio_data.shape}")
            print(f"[DEBUG] Audio min/max: {audio_data.min():.4f}/{audio_data.max():.4f}")
            print(f"[DEBUG] Audio mean/std: {np.mean(audio_data):.4f}/{np.std(audio_data):.4f}")
            
            # Test audio encoder
            audio_embeds = self.audio_encoder.audio2feat(audio_path)
            print(f"[DEBUG] Audio embeddings (raw output) shape: {audio_embeds.shape if hasattr(audio_embeds, 'shape') else type(audio_embeds)}")
            if hasattr(audio_embeds, 'shape'):
                print(f"[DEBUG] Audio embeddings min/max: {audio_embeds.min():.4f}/{audio_embeds.max():.4f}")
                print(f"[DEBUG] Audio embeddings mean/std: {audio_embeds.mean():.4f}/{audio_embeds.std():.4f}")
                print(f"[DEBUG] Audio embeddings dtype: {audio_embeds.dtype}")
            
            # Test feature2chunks
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=audio_embeds, fps=25)
            print(f"\n[DEBUG] Whisper chunks count: {len(whisper_chunks)}")
            if len(whisper_chunks) > 0:
                print(f"[DEBUG] First chunk shape: {whisper_chunks[0].shape}")
                print(f"[DEBUG] First chunk min/max: {whisper_chunks[0].min():.4f}/{whisper_chunks[0].max():.4f}")
                print(f"[DEBUG] First chunk mean/std: {whisper_chunks[0].mean():.4f}/{whisper_chunks[0].std():.4f}")
                
                # Check total expected dimensions for UNet cross attention
                print(f"\n[DEBUG] Audio encoder config:")
                print(f"[DEBUG]   embedding_dim: {self.audio_encoder.embedding_dim}")
                print(f"[DEBUG]   num_frames: {self.audio_encoder.num_frames}")
                
                # Stack 16 chunks like pipeline does
                test_stacked = torch.stack(whisper_chunks[:16])
                print(f"\n[DEBUG] Stacked 16 chunks shape (for UNet): {test_stacked.shape}")
                
        except Exception as e:
            print(f"[DEBUG] Audio encoder test error: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n[DEBUG] Running pipeline...")
        
        # Run inference - match HuggingFace implementation
        self.pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_path,
            num_frames=self.config.data.num_frames,
            num_inference_steps=self.config.run.inference_steps,
            guidance_scale=1.0,  # HuggingFace uses 1.0
            weight_dtype=self.dtype,
            width=self.config.data.resolution,
            height=self.config.data.resolution,
            mask=self.config.data.mask,  # fix_mask
        )
        
        print(f"\n[DEBUG] Pipeline completed.")
        print(f"[DEBUG] Output file exists: {os.path.exists(output_path)}")
        if os.path.exists(output_path):
            print(f"[DEBUG] Output file size: {os.path.getsize(output_path)} bytes")
        
        # Read output
        with open(output_path, "rb") as f:
            out_bytes = f.read()
            
        # Cleanup
        os.remove(video_path)
        os.remove(audio_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        print("="*60)
        print("INFERENCE COMPLETE")
        print("="*60)
            
        return out_bytes

if __name__ == "__main__":
    # Local entrypoint for testing
    pass
