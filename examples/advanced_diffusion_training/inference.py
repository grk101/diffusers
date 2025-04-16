import torch
import argparse
import os
from diffusers import AutoPipelineForText2Image
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the base model (local or HF Hub)")
parser.add_argument("--lora_path", type=str, required=True, help="Path to the folder containing LoRA weights")
parser.add_argument("--token", type=str, required=True, help="The special token used during training")
parser.add_argument("--use_t5", action="store_true", help="Enable if you trained the T5 encoder embedding")
args = parser.parse_args()

# 1. Load the base Flux pipeline
pipe = AutoPipelineForText2Image.from_pretrained(
    args.model_path, torch_dtype=torch.float16
).to("cuda")

# 2. Load the LoRA weights into the pipeline
#pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors")

# 3. Load the learned text embedding for the new token
embedding_path = os.path.join(args.lora_path, f"{os.path.dirname(args.lora_path)}_emb.safetensors")
if os.path.exists(embedding_path):
    state_dict = load_file(embedding_path)
    if "clip_l" in state_dict:
        pipe.load_textual_inversion(state_dict["clip_l"], token=args.token, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    if args.use_t5 and "t5" in state_dict:
        pipe.load_textual_inversion(state_dict["t5"], token=args.token, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
else:
    print(f"[WARN] Embedding file not found: {embedding_path}. Skipping textual inversion.")


prompt = f"a photograph of {args.token} in medieval armor, cinematic lighting"
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("result.png")
