#!/usr/bin/env python3
import os
import glob
from PIL import Image
import argparse

def create_gif(input_dir, output_filename, duration=100):
    """
    Create a GIF from PNG images in the specified directory.
    
    Args:
        input_dir: Directory containing PNG images
        output_filename: Output GIF filename
        duration: Duration of each frame in milliseconds
    """
    # Get all PNG files in the directory
    png_files = glob.glob(os.path.join(input_dir, "render_*.png"))
    
    # Sort files naturally (so render_10000.png comes after render_9999.png, not after render_1.png)
    png_files.sort(key=lambda x: int(os.path.basename(x).replace("render_", "").replace(".png", "")))
    
    print(f"Found {len(png_files)} PNG files in {input_dir}")
    
    # Load all images
    images = []
    for png_file in png_files:
        img = Image.open(png_file)
        images.append(img)
    
    # Save as GIF
    if images:
        images[0].save(
            output_filename,
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=duration,
            loop=0  # 0 means loop forever
        )
        print(f"GIF created: {output_filename}")
    else:
        print("No images found to create GIF")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Create a GIF from PNG files in a directory.')
    parser.add_argument('--dir', '-d', type=str, default=os.getcwd(),
                        help='Directory containing PNG files (default: current directory)')
    parser.add_argument('--output', '-o', type=str, default='animation.gif',
                        help='Output GIF filename (default: animation.gif)')
    parser.add_argument('--duration', type=int, default=100,
                        help='Duration of each frame in milliseconds (default: 100)')
    
    args = parser.parse_args()
    
    # Create GIF from PNG files
    create_gif(args.dir, args.output, args.duration) 