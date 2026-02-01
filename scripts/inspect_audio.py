
import argparse
import librosa
import numpy as np
import os
import sys

def inspect_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return

    print(f"Inspecting audio file: {audio_path}")
    
    try:
        # Load audio with librosa (standard method)
        y, sr = librosa.load(audio_path, sr=16000)
        
        print(f"-"*30)
        print(f"Properties:")
        print(f"  Sample Rate: {sr}")
        print(f"  Duration: {len(y)/sr:.2f} seconds")
        print(f"  Samples: {len(y)}")
        print(f"  Shape: {y.shape}")
        
        print(f"-"*30)
        print(f"Statistics:")
        print(f"  Min: {y.min():.4f}")
        print(f"  Max: {y.max():.4f}")
        print(f"  Mean: {np.mean(y):.4f}")
        print(f"  Std Dev: {np.std(y):.4f}")
        print(f"  RMS: {np.sqrt(np.mean(y**2)):.4f}")
        
        # Check for silence/anomalies
        if np.max(np.abs(y)) < 0.01:
            print(f"\n[WARNING] Audio seems very quiet or silent!")
            
        print(f"-"*30)
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect audio file properties for LatentSync debugging")
    parser.add_argument("audio_path", help="Path to the audio file")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    inspect_audio(args.audio_path)
