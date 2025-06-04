import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_progress_grid(image_folder, output_path=None, columns=3, image_size=(512, 512), padding=10, bg_color=(255, 255, 255)):
    """
    Create a grid visualization of generated images from a folder.
    
    Args:
        image_folder (str): Path to the folder containing generated images
        output_path (str, optional): Path to save the output image. Defaults to 'progress_grid.png' in the image folder.
        columns (int, optional): Number of columns in the grid. Defaults to 3.
        image_size (tuple, optional): Size to which each image will be resized. Defaults to (512, 512).
        padding (int, optional): Padding between images in pixels. Defaults to 10.
        bg_color (tuple, optional): Background color in RGB. Defaults to white (255, 255, 255).
    
    Returns:
        str: Path to the saved grid image
    """
    # Convert to Path objects
    image_folder = Path(image_folder)
    if output_path is None:
        output_path = image_folder / 'progress_grid.png'
    else:
        output_path = Path(output_path)
    
    # Get all image files from the folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = sorted(
        [f for f in image_folder.glob('*') if f.suffix.lower() in image_extensions],
        key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
    )
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    
    num_images = len(image_files)
    rows = (num_images + columns - 1) // columns
    
    # Calculate total size of the grid
    total_width = columns * (image_size[0] + padding) - padding
    total_height = rows * (image_size[1] + padding + 30) - padding  # Extra space for iteration number
    
    # Create a new image with white background
    grid_image = Image.new('RGB', (total_width, total_height), color=bg_color)
    
    # Try to load a font, fall back to default if not available

    font_size = 20
    try:
        # Try to load the default font with specified size
        font = ImageFont.load_default()
        # Scale the default font to the desired size
        font = ImageFont.truetype("arialbd.ttf", font_size) if hasattr(ImageFont, 'truetype') else font
    except:
        # Fallback to default font if loading fails
        font = ImageFont.load_default()
    
    # Paste each image into the grid
    for i, img_path in enumerate(image_files):
        try:
            # Open and resize the image
            img = Image.open(img_path)
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            
            # Calculate position
            x = (i % columns) * (image_size[0] + padding)
            y = (i // columns) * (image_size[1] + padding + 30)  # Extra space for text
            
            # Paste the image
            grid_image.paste(img, (x, y))
            
            # Add iteration number
            draw = ImageDraw.Draw(grid_image)
            iteration_text = f"Iteration {i + 1}"
            text_width = draw.textlength(iteration_text, font=font)
            text_x = x + (image_size[0] - text_width) // 2
            text_y = y + image_size[1] + 5
            draw.text((text_x, text_y), iteration_text, fill=(0, 0, 0), font=font)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save the grid image
    grid_image.save(output_path)
    print(f"Progress grid saved to: {output_path}")
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Create a progress grid of generated images.')
    parser.add_argument('--folder', type=str, default='generated_images',
                       help='Path to the folder containing generated images (default: generated_images)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the grid image (default: <folder>/progress_grid.png)')
    parser.add_argument('--columns', type=int, default=3,
                       help='Number of columns in the grid (default: 3)')
    
    args = parser.parse_args()
    
    try:
        output_path = create_progress_grid(
            image_folder=args.folder,
            output_path=args.output,
            columns=args.columns
        )
        print(f"Successfully created progress grid at: {output_path}")
    except Exception as e:
        print(f"Error creating progress grid: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
