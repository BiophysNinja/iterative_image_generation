import os
import torch
from diffusers import AmusedPipeline, StableDiffusionPipeline, AutoPipelineForText2Image
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Configuration
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_ITERATIONS = 3
def initialize_pipeline():
    """Initialize the image generation pipeline."""
    device = "cuda" if torch.cuda.is_available()  else "cpu"
    print(f"Using device: {device}")
    
    # pipe = AmusedPipeline.from_pretrained(
    #     "amused/amused-512", 
    #     variant="fp16", 
    #     torch_dtype=torch.float16 if device != "cpu" else torch.float32
    # )
    # model_id = "sd-legacy/stable-diffusion-v1-5"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
   
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32)
    return pipe.to(device), device

def generate_image(pipe, prompt, negative_prompt, generator, device):
    """Generate an image using the given prompt and negative prompt."""
    with torch.autocast(device if device != 'cuda' else 'cpu'):  # MPS has issues with autocast
        # Generate the image
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=1,
            guidance_scale=0.0,
        )
        
        # Get the first image from the result
        image = result.images[0]
        
        # If image is a torch.Tensor, convert it to PIL Image
        if hasattr(image, 'cpu'):
            image = image.cpu()
        if hasattr(image, 'numpy'):
            image = Image.fromarray((image.numpy() * 255).astype('uint8'))
        
        return image

def analyze_image_with_Gemma(image, prompt, iteration):
    """Analyze the generated image and suggest improvements using Gemma3."""
    try:
        # Configure Geemma
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemma-3-27b-it')
        
        # Create analysis prompt
        analysis_prompt = f"""
        You are an expert image critic specializing in professional photography and digital media. 
        Analyze this image and provide feedback to help generate award winning photographs:
        
        1. Technical Quality Assessment:
           - Resolution and sharpness for large format printing
           - Color accuracy and dynamic range
           - Noise levels and image artifacts
           - Lighting and exposure balance
        
        2. Composition & Professional Standards:
           - Rule of thirds and visual balance
           - Subject placement and framing
           - Professional aesthetic appeal
           - Suitability for commercial use
        
        3. Specific Improvements Needed:
           - Technical adjustments (exposure, contrast, white balance)
           - Compositional improvements
           - Elements that need refinement or removal
        
        4. Prompt Enhancement Suggestions:
           - Specific, actionable suggestions to improve the prompt for professional results
           - Keywords to enhance image quality and professionalism
           - Style and mood descriptors for a polished look
        
        Current prompt: {prompt}
        Iteration: {iteration + 1}/{NUM_ITERATIONS}
        """
        
        # Generate the analysis
        response = model.generate_content([analysis_prompt, image])
        
        return response.text
        
    except Exception as e:
        print(f"Error analyzing image with Gemma: {e}")
        import traceback
        traceback.print_exc()
        return "Unable to analyze image. Continuing with current prompt."

def improve_prompt(current_prompt, current_negative_prompt, analysis, prompts_history, negative_prompts_history):
    """Use Gemma to improve the prompt based on the analysis."""
    try:
        prompt_hist_str = "\n".join(prompts_history)
        neg_prompt_hist_str = "\n".join(negative_prompts_history)
        
        # Configure Gemma
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemma-3-27b-it')
        
        improvement_prompt = f"""
        Based on this image analysis:
        {analysis}
        
        Current prompt: {current_prompt}
        Current negative prompt: {current_negative_prompt}
        
        Previous prompts history:
        {prompt_hist_str}
        
        Previous negative prompts history:
        {neg_prompt_hist_str}
        
        Please provide:
        1. An improved version of the prompt
        2. An improved version of the negative prompt
        
        Make small changes to the prompts to improve the image.
        
        Format your response as:
        IMPROVED_PROMPT: [your improved prompt]
        NEGATIVE_PROMPT: [your improved negative prompt]
        """
        
        # Generate the improved prompts
        response = model.generate_content(improvement_prompt)
        text = response.text
        
        # Parse the response
        new_prompt = current_prompt
        new_negative_prompt = current_negative_prompt
        
        if "IMPROVED_PROMPT:" in text:
            new_prompt = text.split("IMPROVED_PROMPT:", 1)[1].split("NEGATIVE_PROMPT:", 1)[0].strip()
        if "NEGATIVE_PROMPT:" in text:
            new_negative_prompt = text.split("NEGATIVE_PROMPT:", 1)[1].strip()
            
        return new_prompt, new_negative_prompt
        
    except Exception as e:
        print(f"Error improving prompts: {e}")
        import traceback
        traceback.print_exc()
        return current_prompt, current_negative_prompt

def main():
    # Initialize
    pipe, device = initialize_pipeline()
    generator = torch.Generator(device).manual_seed(42)
    
    # Initial prompt
    prompt = "a small black dog"
    negative_prompt = "low quality, blurry"
    
    prompts_history = []
    negative_prompts_history = []
    prompts_history.append(prompt)
    negative_prompts_history.append(negative_prompt)
    
    for i in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {i + 1}/{NUM_ITERATIONS} ---")
        print(f"Prompt: {prompt}")
        print(f"Negative prompt: {negative_prompt}")
        
        # Generate image
        print("Generating image...")
        image = generate_image(pipe, prompt, negative_prompt, generator, device)
        
        # Save the image
        output_path = OUTPUT_DIR / f"iteration_{i+1}.png"
        image.save(output_path) 
        print(f"Saved image to {output_path}")
        
        # Get analysis from Gemma
        print("Analyzing image with Gemma...")
        analysis = analyze_image_with_Gemma(image, prompt, i)
        print("\nAnalysis:", analysis)
        
        # Improve prompts for next iteration
        if i < NUM_ITERATIONS - 1:  # No need to improve on last iteration
            print("\nImproving prompts...")
            prompt, negative_prompt = improve_prompt(prompt, negative_prompt, analysis, prompts_history, negative_prompts_history)
            print(f"New prompt: {prompt}")
            print(f"New negative prompt: {negative_prompt}")
            prompts_history.append(prompt)
            negative_prompts_history.append(negative_prompt)
    
    print("\nImage generation complete! Check the 'generated_images' folder for results.")

if __name__ == "__main__":
    main()
