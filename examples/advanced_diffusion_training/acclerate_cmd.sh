#!/usr/bin/bash 

 accelerate launch train_dreambooth_lora_flux_advanced.py \
	 --pretrained_model_name_or_path=$MODEL_NAME \
	 --instance_data_dir=$INSTANCE_DIR \
	 --output_dir=$OUTPUT_DIR \
	 --instance_prompt="a photo of TOK person" \
	 --resolution=512 \
	 --center_crop \
	 --train_batch_size=1 \
	 --max_train_steps=500 \
	 --gradient_checkpointing \
	 --checkpointing_steps=500 \
	 --lr_scheduler="constant" --lr_warmup_steps=0 \
	 --rank=16 \
	 --repeats=1 \
	 --mixed_precision="fp16" \
	 --optimizer="AdamW" \
	 --use_8bit_adam \
	 --learning_rate=5e-4 \
	 --train_text_encoder_ti \
	 --lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2"




   

