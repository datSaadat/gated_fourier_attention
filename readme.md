# Gated Fourier–Attention RoBERTa

This repository contains a modified RoBERTa implementation where each Transformer layer can choose between **self-attention** and a **Fourier token mixer** via a learnable **Gumbel–Softmax gate**. The model is trained end-to-end and used to study when Transformers prefer attention over spectral mixing.

---

## 1. Code Structure

### Modified `transformers` source  
The file `modeling_roberta.py` is patched to include:
- Fourier mixer (2D FFT)
- Gumbel–Softmax gate
- Learned mixing coefficients
- DropPath support  
Place this file inside your local `transformers/models/roberta/` directory.

### Training script  
Use `run_mlm_gated.py` (a lightly modified version of HF's MLM trainer) to launch pretraining.

---

## 2. Training Example

Run distributed training with:

```bash
torchrun --nproc_per_node=2 run_mlm_gated.py \
  --dataset_name JackBAI/bert_pretrain_datasets \
  --config_name "/Data/github/RoBERTa/RoBERTa-Pretrain_gated/configs/roberta_medium.json" \
  --tokenizer_name "$TOKENIZER" \
  --do_train \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_seq_length 512 \
  --mlm_probability 0.15 \
  --logging_steps 100 \
  --save_steps 2000 \
  --save_total_limit 2 \
  --report_to none \
  --output_dir "$OUT" \
  --overwrite_output_dir \
  --bf16 \
  > new_run_log_gated.txt
