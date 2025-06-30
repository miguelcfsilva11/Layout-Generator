accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --train_data_dir="./train_data" \
  --output_dir="./fantasy_map_lora" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=10 \
  --learning_rate=1e-4 \
  --report_to="none" \
  --caption_column="text"