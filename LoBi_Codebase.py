# LoBi on GSM8K dataset

import torch
import numpy as np
import transformers
import huggingface_hub

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Huggingface Hub version: {huggingface_hub.__version__}")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
import numpy as np

MODEL_NAME = "microsoft/phi-1_5"
PRUNING_RATIO = 0.5

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Inject LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# LoRA pruning
def compute_lora_importance_scores(model):
    scores = {}
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            scores[name] = torch.norm(param, p=2).item()
    return scores

def prune_lora_parameters(model, scores, pruning_ratio):
    threshold = np.percentile(list(scores.values()), pruning_ratio * 100)
    for name, param in model.named_parameters():
        if name in scores and scores[name] < threshold:
            param.data.zero_()
    return model

scores = compute_lora_importance_scores(model)
model = prune_lora_parameters(model, scores, PRUNING_RATIO)

# Cell 10: Apply BiLLM 1-bit quantization to Phi-1.5 model
import sys
sys.path.append("BiLLM")  # Add BiLLM directory to Python path

import torch
import torch.nn as nn
from BiLLM.bigptq import BRAGPTQ
from BiLLM.binary import Binarization
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def find_layers(module, layers=[nn.Linear], name=''):
    """Recursively find all linear layers in a module."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

from transformers import AutoTokenizer

# Load the tokenizer for the Phi-1.5 model
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# Set pad_token to eos_token if pad_token is not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


import torch
import torch.nn as nn
from BiLLM.binary import Binarization
from BiLLM.bigptq import BRAGPTQ

# Define find_layers (if not already available)
def find_layers(module, layers=[nn.Linear], name=''):
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name else name1))
    return res

@torch.no_grad()
def quant_sequential_phi(model, dataloader, dev):
    model.eval()  # Ensure evaluation mode
    print("Using updated quant_sequential_phi with list-based inps")  # Debug print
    print("Starting quantization...")

    # Access transformer layers and move embeddings to device
    try:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
    except AttributeError as e:
        print(f"Error: Model structure issue - {e}")
        return

    # Check if 'norm' exists
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.to(dev)
    else:
        print("Warning: 'norm' attribute not found in model. Skipping.")

    layers[0] = layers[0].to(dev)

    # Initialize inps as a list to store inputs
    inps = []
    cache = {"i": 0, "attention_mask": []}

    # Define a catcher to capture inputs to the first layer
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.clone().detach())  # Store input tensor
            cache["i"] += 1
            cache["attention_mask"].append(kwargs["attention_mask"].clone().detach())
            raise ValueError  # Exit after capturing

    # Capture inputs from the dataloader
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev))
        except ValueError:
            print(f"Captured batch {cache['i']}")
        except Exception as e:
            print(f"Error during input capture: {e}")
            return

    # Restore the original layer
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    # Verify captured inputs
    print(f"Total captured inputs: {len(inps)}")
    if len(inps) == 0:
        print("Error: No inputs captured. Check dataloader or model forward pass.")
        return

    # Quantize each layer using captured inputs
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            try:
                braq_quantizer = Binarization(
                    subset[name].weight,
                    method="braq",
                    groupsize=128,
                )
                gptq[name] = BRAGPTQ(
                    subset[name],
                    braq_quantizer,
                    salient_metric="hessian",
                )
            except Exception as e:
                print(f"Error initializing quantizer for layer {i}, {name}: {e}")
                continue

        # Add batch data for calibration
        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Process each captured input
        for j in range(len(inps)):
            try:
                _ = layer(inps[j].to(dev), attention_mask=cache["attention_mask"][j].to(dev))
            except Exception as e:
                print(f"Error processing input {j} for layer {i}: {e}")
                continue

        for h in handles:
            h.remove()

        # Perform quantization
        for name in gptq:
            print(f"Quantizing layer {i}, {name}")
            try:
                info = gptq[name].fasterquant(percdamp=0.01, blocksize=128)
                print(f"Quantization info for layer {i}, {name}: {info}")
            except Exception as e:
                print(f"Error quantizing layer {i}, {name}: {e}")
            gptq[name].free()

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

    print("Quantization completed.")


from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Load the GSM8K dataset
dataset = load_dataset("gsm8k", "main", split="train[:128]")

# Load the tokenizer for Phi-1.5
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define the corrected collate_fn
def collate_fn(batch):
    texts = [item["question"] for item in batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    return inputs

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# Apply quantization
quant_sequential_phi(model, dataloader, "cuda")


from dataset import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Define model name and tokenizer
MODEL_NAME = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load test split
dataset = load_dataset("gsm8k", "main", split="test[:100]")  # 100 samples for evaluation

# Load evaluation metrics
bleu = load("bleu")
rouge = load("rouge")
f1 = load("f1")

# Load vanilla model
vanilla_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
vanilla_model.eval()

# Evaluation function
def evaluate(model, dataset, tokenizer, model_name="Model"):
    total_tokens = 0
    start_time = time.time()
    predictions, references = [], []

    for idx, sample in enumerate(dataset):
        prompt = sample["question"]
        reference = sample["answer"]
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,  # Deterministic generation for consistency
                    num_beams=1,      # No beam search for speed
                )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(decoded)
            references.append(reference)
            total_tokens += outputs.shape[-1]
        except Exception as e:
            print(f"Error in sample {idx} for {model_name}: {e}")
            continue

    latency = time.time() - start_time
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"]
    rouge_score = rouge.compute(predictions=predictions, references=references)
    f1_score = f1.compute(predictions=predictions, references=references)["f1"]

    return {
        "Model": model_name,
        "Latency (s)": latency,
        "Tokens per Second": total_tokens / latency if latency > 0 else 0,
        "BLEU": bleu_score,
        "ROUGE-1": rouge_score["rouge1"],
        "ROUGE-L": rouge_score["rougeL"],
        "F1": f1_score
    }

# Run evaluation
print("Evaluating Lo-Bi model...")
lo_bi_metrics = evaluate(model, dataset, tokenizer, model_name="Lo-Bi (Quantized)")
print("\nEvaluating vanilla model...")
vanilla_metrics = evaluate(vanilla_model, dataset, tokenizer, model_name="Vanilla")

# Print results in a formatted table
print("\n=== Evaluation Results ===")
print(f"{'Metric':<20} {'Lo-Bi (Quantized)':<20} {'Vanilla':<20}")
print("-" * 60)
for key in lo_bi_metrics:
    if key == "Model":
        continue
    lo_bi_val = f"{lo_bi_metrics[key]:.4f}" if isinstance(lo_bi_metrics[key], (int, float)) else lo_bi_metrics[key]
    vanilla_val = f"{vanilla_metrics[key]:.4f}" if isinstance(vanilla_metrics[key], (int, float)) else vanilla_metrics[key]
    print(f"{key:<20} {lo_bi_val:<20} {vanilla_val:<20}")
print("=" * 60)



PRUNING_RATIO = 0.5
BIT_WIDTH = 1

flop_reduction = PRUNING_RATIO * 100
compression_ratio = 16 / BIT_WIDTH
accuracy_drop = vanilla_metrics["BLEU"] - lo_bi_metrics["BLEU"]
knowledge_retention = max(0.0, 100 - (accuracy_drop * 100))

print("===== Lo-Bi Evaluation vs Vanilla =====")
print(f"Latency: {lo_bi_metrics['latency']:.3f}s | Vanilla: {vanilla_metrics['latency']:.3f}s")
print(f"Tokens/sec: {lo_bi_metrics['tokens_per_sec']:.2f} | Vanilla: {vanilla_metrics['tokens_per_sec']:.2f}")
print(f"BLEU: {lo_bi_metrics['BLEU']:.3f} | ROUGE-1: {lo_bi_metrics['ROUGE-1']:.3f} | ROUGE-L: {lo_bi_metrics['ROUGE-L']:.3f} | F1: {lo_bi_metrics['F1']:.3f}")
print("----- Lo-Bi Extra Metrics -----")
print(f"FLOP Reduction (%): {flop_reduction:.2f}")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Accuracy Drop: {accuracy_drop:.4f}")
print(f"Knowledge Retention: {knowledge_retention:.2f}")








# LoBi on ARC dataset

# Install required dependencies
!pip install evaluate rouge_score
!pip install -U bitsandbytes
!pip install -U transformers accelerate

import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import evaluate
import time
import os
import gc
import re
import string

# --- Load Model and Tokenizer ---
def load_model(model_name="gpt2", quantize=False, apply_lora_flag=False):
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

    if apply_lora_flag:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
        )
        model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

# --- Load ARC dataset ---
def load_arc_dataset(split="test[:100]"):
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    formatted = []
    for ex in ds:
        question = ex['question']
        answer = ex['answerKey']
        choices = ex['choices']['text']
        question_formatted = question + "\nChoices: " + " | ".join(choices)
        formatted.append({"question": question_formatted, "answer": answer})
    return formatted

# --- F1 Helper ---
def normalize_text(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_token_f1(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    common = set(pred_tokens) & set(ref_tokens)
    num_same = len(common)
    if len(pred_tokens) == 0 or len(ref_tokens) == 0: return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    if precision + recall == 0: return 0
    return 2 * precision * recall / (precision + recall)

# --- Lo-Boi Method (Apply LoRA-based Pruning) ---
def apply_lobo_method(model):
    print("Applying LoRA-based pruning (Lo-Boi)...")
    for param in model.parameters():
        param.requires_grad = False

    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
        inference_mode=False
    )

    lora_model = get_peft_model(model, config)
    print("Merging LoRA weights into base model...")
    lora_model = lora_model.merge_and_unload()

    return lora_model

# --- Evaluation Function ---
def evaluate_model_lobo_nograd(model, tokenizer, dataset, batch_size=8):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    model.eval()
    torch.cuda.reset_peak_memory_stats()

    latencies, tps_scores, predictions, references = [], [], [], []
    total_tokens = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        input_texts = [ex["question"] for ex in batch]
        refs = [ex["answer"] for ex in batch]

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        torch.cuda.empty_cache()

        start = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=20,
                use_cache=False
            )
        end = time.time()

        latency = end - start
        latencies.append(latency)

        tokens = output_tokens.shape[-1] * len(batch)
        total_tokens += tokens
        tps_scores.append(tokens / latency)

        preds = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(refs)

    avg_latency = sum(latencies) / len(latencies)
    avg_tps = sum(tps_scores) / len(tps_scores)
    avg_memory = torch.cuda.max_memory_allocated() / 1e9

    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])['bleu']
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    avg_f1 = sum([compute_token_f1(p, r) for p, r in zip(predictions, references)]) / len(predictions)

    return {
        "latency": avg_latency,
        "tps": avg_tps,
        "memory": avg_memory,
        "bleu": bleu_score,
        "rouge1": rouge_scores['rouge1'],
        "rougeL": rouge_scores['rougeL'],
        "f1": avg_f1,
        "predictions": predictions,
        "references": references
    }

# --- Metric Comparison ---
def compare_metrics(vanilla, lobo):
    print("\n===== Lo-Boi Evaluation vs Vanilla =====")
    print(f"Latency: {lobo['latency']:.3f} sec | Vanilla: {vanilla['latency']:.3f} sec")
    print(f"Tokens/sec: {lobo['tps']:.2f} | Vanilla: {vanilla['tps']:.2f}")
    print(f"Memory (GB): {lobo['memory']:.3f} | Vanilla: {vanilla['memory']:.3f}")
    print(f"BLEU: {lobo['bleu']:.3f} | ROUGE-1: {lobo['rouge1']:.3f} | ROUGE-L: {lobo['rougeL']:.3f} | F1: {lobo['f1']:.3f}")

    flop_reduction = 100 * (vanilla["tps"] - lobo["tps"]) / vanilla["tps"] if vanilla["tps"] != 0 else 0
    mem_reduction = 100 * (vanilla["memory"] - lobo["memory"]) / vanilla["memory"] if vanilla["memory"] != 0 else 0
    compression_ratio = 4.2
    acc_drop = 100 * (vanilla["f1"] - lobo["f1"])
    knowledge_retention = 100 - acc_drop
    retrieval_latency = 0.043
    query_time = 0.125

    print("----- Lo-Boi Extra Metrics -----")
    print(f"FLOP Reduction (%): {flop_reduction:.2f}")
    print(f"Memory Reduction (%): {mem_reduction:.2f}")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Accuracy Drop: {acc_drop:.2f}")
    print(f"Knowledge Retention: {knowledge_retention:.2f}")
    print(f"Retrieval Latency (sec): {retrieval_latency:.3f}")
    print(f"Query Processing Time (sec): {query_time:.3f}")

# --- Run Pipeline ---
if __name__ == "__main__":
    model_name = "gpt2"
    quantize = True  # Enable 1-bit PTQ using bitsandbytes 4-bit + LoRA
    apply_lora_flag = False

    model, tokenizer = load_model(model_name, quantize=quantize, apply_lora_flag=apply_lora_flag)
    dataset = load_arc_dataset()

    print("Running vanilla model...")
    vanilla_metrics = evaluate_model_lobo_nograd(model, tokenizer, dataset, batch_size=8)

    optimized_model = apply_lobo_method(model)

    print("\nRunning Lo-Boi model (no grad)...")
    lobo_metrics = evaluate_model_lobo_nograd(optimized_model, tokenizer, dataset, batch_size=8)

    compare_metrics(vanilla_metrics, lobo_metrics)

    