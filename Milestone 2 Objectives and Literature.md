## Objectives 

Our team’s goal is to explore how generative AI can be used to create novel artistic styles by blending influences from multiple iconic artists. In this phase of the project, we are using Stable Diffusion to generate paintings conditioned on artist-specific prompts. We want to understand how different text inputs influence the visual characteristics of the output, and this serves as a first step toward our broader goal of generating entirely new, blended painting styles. 

Our objectives are: 

1. Generate Stylized Artworks Using Artist-Conditioned Prompts 
2. Use a pre-trained Stable Diffusion model to create artwork based on prompts that include artist names 
3. Understand Prompt-Conditioning Behavior 

Analyze how well different prompts (e.g., "in the style of Van Gogh") influence the generated image outputs. This includes controlling randomness through seeding and comparing results across artists. 
4. Train Lightweight Artist-Specific Style Modules with LoRA 

We fine-tune Stable Diffusion for individual artists using LoRA, allowing us to adapt the model to each artist’s unique style without retraining the entire model. 
5. Enable Controlled Style Fusion by Merging LoRA Adapters 

After training LoRA modules for several artists, we merge them using custom weights to blend their styles in new ways. This lets users generate hybrid-style images like “60% Picasso, 40% Van Gogh.” 

## Literature Review 

We based our work on recent research and tools at the intersection of natural language processing and generative image modeling: 

Stable Diffusion (Rombach et al., 2022): 
- This is the backbone of our current model. Stable Diffusion is a latent diffusion model that uses a CLIP-based text encoder to condition image generation on natural language prompts. It has shown impressive capabilities in style adherence and prompt alignment, which is why we chose it for this phase of the project. 

CLIP (Radford et al., 2021):
- Although we’re using it indirectly through Stable Diffusion, CLIP plays a crucial role in aligning textual and visual representations. It allows the model to understand the semantic meaning behind prompts like "in the style of Picasso" or "impressionist landscape." 

LoRA (Hu et al., 2021):
- LoRA enables parameter-efficient fine-tuning by injecting small, trainable rank-decomposed matrices into existing models. It allows us to fine-tune Stable Diffusion on a per-artist basis without modifying the full model weights. This makes it possible to personalize and modularize style representations efficiently.  