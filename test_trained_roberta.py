# test_trained_roberta.py  (robust eval for MLM checkpoints)
import os, math, argparse, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, RobertaForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

# Silence HF tokenizer fork warnings proactively.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TEXT_KEYS = ["text","content","article","document","story","body","headline","title","summary","description"]

def coerce_text(example):
    """
    Normalize any dataset to a single 'text' string:
    - If a suitable string field exists, use it
    - If it is a list/sequence of strings, join with spaces
    - Else return empty string (will be filtered out)
    """
    v = example.get("text", None)
    if isinstance(v, str) and v.strip():
        return {"text": v}
    for k in TEXT_KEYS:
        if k in example:
            val = example[k]
            if isinstance(val, str) and val.strip():
                return {"text": val}
            if isinstance(val, (list, tuple)):
                # join list of strings, ignoring non-strings
                joined = " ".join(x for x in val if isinstance(x, str))
                return {"text": joined}
    return {"text": ""}

def parse_spec(spec: str):
    """
    Accepts:
      - name|split
      - name|config|split
    Examples:
      wikitext|wikitext-2-raw-v1|test
      cc_news|train[:2%]
    """
    parts = spec.split("|")
    if len(parts) == 2:
        name, split = parts
        cfg = None
    elif len(parts) == 3:
        name, cfg, split = parts
    else:
        raise ValueError(f"Bad dataset spec '{spec}'. Use 'name|split' or 'name|config|split'.")
    return name, (cfg if len(parts) == 3 else None), split

def load_named_split(spec, max_samples=None, seed=42):
    name, cfg, split = parse_spec(spec)
    ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
    # Keep only columns we might need to construct 'text'; then coerce to 'text'
    keep_cols = [c for c in ds.column_names if c in set(TEXT_KEYS + ["text"])]
    if keep_cols and set(keep_cols) != set(ds.column_names):
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    ds = ds.map(coerce_text, remove_columns=[c for c in ds.column_names if c != "text"])
    ds = ds.filter(lambda e: isinstance(e["text"], str) and len(e["text"].strip()) > 0)
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds

def tokenize_and_strip(tokenizer, max_len=512):
    def _tok(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len)
        return enc
    return _tok

def eval_one(model, tokenizer, ds, batch_size=16, workers=0, fp16=False):
    # Tokenize and DROP everything else so the collator only sees model tensors
    ds_tok = ds.map(tokenize_and_strip(tokenizer, 512), batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir="./_tmp_eval",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=workers,  # keep 0 for robustness with tokenizers
        fp16=fp16,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, data_collator=collator)
    metrics = trainer.evaluate(ds_tok)
    loss = float(metrics["eval_loss"])
    ppl = math.exp(loss) if loss < 50 else float("inf")

    # Masked-token accuracy using the same collator masking
    from torch.utils.data import DataLoader
    dl = DataLoader(ds_tok, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=workers)
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits
            preds = logits.argmax(-1)
            mask = batch["labels"] != -100
            correct += (preds[mask] == batch["labels"][mask]).sum().item()
            total += mask.sum().item()
    acc = (correct / total) if total > 0 else 0.0
    return loss, ppl, acc, len(ds_tok)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Specs: 'name|split' or 'name|config|split'. Supports slices, e.g., train[:2%].")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)  # default 0 for stability
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    print(f"ckpt={args.ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = RobertaForMaskedLM.from_pretrained(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for spec in args.datasets:
        try:
            ds = load_named_split(spec, max_samples=args.max_samples)
            loss, ppl, acc, n = eval_one(model, tokenizer, ds,
                                         batch_size=args.batch_size,
                                         workers=args.workers,
                                         fp16=args.fp16)
            print(f"{spec:35s} | loss={loss:.3f} | ppl={ppl:.2f} | masked_acc={acc:.4f} | n={n:,}")
        except Exception as e:
            print(f"{spec:35s} | ERROR: {e}")

if __name__ == "__main__":
    main()

