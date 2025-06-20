# Iterative Image Generation with VLM Feedback

This project demonstrates an iterative image generation pipeline that uses a Vision Language Model (VLM) to provide feedback and improve generated images over multiple iterations. The system generates images using a diffusion model, analyzes them using Google's Gemma3 model, and iteratively refines the prompts to achieve better results.

## Features

- **Iterative Image Generation**: Generate and refine images over multiple iterations
- **AI-Powered Feedback**: Uses Google's Gemini model to analyze and provide feedback on generated images
- **Prompt Optimization**: Automatically improves prompts based on AI feedback
- **Progress Visualization**: Generate a grid visualization of the image generation progress
- **Flexible Configuration**: Customize the number of iterations, image size, and other parameters

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- A Google API key with access to the Gemini API
- CUDA-compatible GPU (recommended for faster generation)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BiophysNinja/iterative_image_generation.git
   cd iterative_image_generation_with_llms
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### 1. Generate Images Iteratively

Run the main script to start the iterative image generation process:

```bash
python iterative_image_generation.py
```

By default, this will:
- Generate 5 iterations of images
- Save them to the `generated_images` directory
- Use the initial prompt "cowboy portrait"

### 2. Visualize the Progress

After generating images, create a progress grid visualization:

```bash
python visualize_progress.py --folder PATH_TO_ITERATION_IMAGE_RESULTS
```

This will create a `progress_grid.png` file in the same directory as your generated images.

## Project Structure

- `iterative_image_generation.py`: Main script for generating images iteratively
- `visualize_progress.py`: Script for creating a grid visualization of generated images
- `generated_images/`: Default directory for storing generated images
- `.env`: Configuration file for API keys (not included in version control)

## Requirements

- torch
- diffusers
- google-generativeai
- python-dotenv
- Pillow
- numpy

## License



## Acknowledgments

- Uses the AMUSED model for image generation
- Utilizes Google's Gemini for image analysis and feedback
- Built with PyTorch and the Hugging Face ecosystem
