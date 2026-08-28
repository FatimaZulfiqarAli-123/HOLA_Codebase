#AdaComp on GSM8K

!pip install datasets
!pip install nltk
!pip install rouge-score
!pip install scikit-learn
!pip install torch
!pip install psutil
!pip install evaluate
!pip install huggingface_hub
!pip install tqdm
!pip install numpy
!pip uninstall -y bitsandbytes
!pip install bitsandbytes==0.43.3
!pip install --no-cache-dir faiss-cpu
!pip install transformers==4.47.0
!pip install sentence-transformers==3.1.1

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)

import subprocess
import sys
import torch
import numpy as np
import re
import gc
import psutil
import time
import os
import json
from typing import List, Dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from sentence_transformers import SentenceTransformer
import faiss

# Verify dependencies
try:
    import bitsandbytes
    from bitsandbytes import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    print(f"Bitsandbytes version: {bitsandbytes.__version__}")
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    if not cuda_version:
        print("Warning: CUDA not detected. Bitsandbytes may not work correctly.")
        BITSANDBYTES_AVAILABLE = False
except ImportError:
    print("Warning: bitsandbytes not installed. Falling back to float16.")
    BITSANDBYTES_AVAILABLE = False

import transformers
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Hugging Face login
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(hf_token)
    print("Successfully logged in to HuggingFace.")
except Exception as e:
    print(f"Error accessing HF_TOKEN: {e}")
    raise Exception("Authentication failed. Please set HF_TOKEN in Kaggle Secrets.")

# Set PyTorch environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuration
model_name = "google/gemma-2b"
dataset_name = "gsm8k"
max_samples = 100
batch_size = 1
max_length = 64
max_new_tokens = 150
embedding_dim = 384
top_k = 3

# Use accelerator device
device = accelerator.device
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("Running on CPU")

# Dataset Loading and Preprocessing
def preprocess_gsm8k(examples):
    prompts = [f"""Solve this math problem step-by-step. Show your reasoning clearly and concisely. Include all relevant numbers and terms from the question in your answer. End with exactly this format: **Final Answer: [number]**

Question: {q}
Answer: """ for q in examples["question"]]
    refs = [a.strip().split("####")[-1].strip() for a in examples["answer"]]
    contexts = [a.strip().split("####")[0].strip() for a in examples["answer"]]
    return {"prompts": prompts, "references": refs, "contexts": contexts}

print("Loading dataset...")
datasets = {
    "gsm8k": load_dataset("gsm8k", "main", split="test").shuffle(seed=42).select(range(max_samples))
}
data = {
    "gsm8k": preprocess_gsm8k(datasets["gsm8k"])
}

# VRAM logging
def log_vram_usage(step: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"{step} - VRAM Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")

# Clear GPU memory
def clear_gpu_memory():
    for _ in range(2):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        log_vram_usage("After GPU memory clear")

# Model & Tokenizer Loading
def load_model_and_tokenizer(model_name, quantize=True):
    log_vram_usage(f"Before loading {model_name}")
    clear_gpu_memory()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if BITSANDBYTES_AVAILABLE and quantize:
            print(f"Using 4-bit quantization for {model_name}...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            model.gradient_checkpointing_enable()
        else:
            print(f"Using float16 for {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        log_vram_usage(f"After loading {model_name}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        raise Exception("Failed to load LLM.")

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"Embedding model loaded on {device}")

# Load LLM
print(f"Loading LLM ({model_name})...")
model, tokenizer = load_model_and_tokenizer(model_name, quantize=True)
model, tokenizer = accelerator.prepare(model, tokenizer)
model_device = next(model.parameters()).device
print(f"Model is on device: {model_device}")

# Build FAISS index for contexts
def build_faiss_index(contexts: List[str], embedder, dim: int) -> faiss.IndexFlatL2:
    embeddings = embedder.encode(contexts, convert_to_numpy=True, show_progress_bar=True, device=model_device)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

faiss_indices = {}
print(f"Building FAISS index for {dataset_name}...")
index, embeddings = build_faiss_index(data[dataset_name]["contexts"], embedder, embedding_dim)
faiss_indices[dataset_name] = {"index": index, "embeddings": embeddings}

# Knowledge Retention Function
def compute_knowledge_retention(source: str, generated: str) -> float:
    """Compute knowledge retention as the percentage of exact numerical matches."""
    if not generated or not source:
        print(f"Debug: Empty source or generated text (source: {source[:50]}..., generated: {generated[:50]}...)")
        return 0.0
    source_numbers = set(re.findall(r'[-+]?\d*\.?\d+', source))
    generated_numbers = set(re.findall(r'[-+]?\d*\.?\d+', generated))
    if not source_numbers:
        print(f"Debug: No numbers extracted from source: {source[:50]}...")
        return 0.0
    matches = len(source_numbers.intersection(generated_numbers))
    retention = matches / len(source_numbers)
    print(f"Debug: Source numbers: {source_numbers}, Generated numbers: {generated_numbers}, Matches: {matches}, Retention: {retention:.3f}")
    return retention * 100  # Convert to percentage

# AdaComp-RAG Functions
def query_classifier(query: str, embedder) -> str:
    tokens = len(query.split())
    has_numbers = bool(re.search(r'\d+', query))
    if tokens < 20 or has_numbers:
        return "direct"
    return "retrieve"

def retrieve_passages(query: str, dataset_name: str, embedder, index: faiss.IndexFlatL2, contexts: List[str], top_k: int = 3) -> List[str]:
    query_emb = embedder.encode([query], convert_to_numpy=True, device=model_device)
    distances, indices = index.search(query_emb, top_k)
    retrieved = [contexts[idx] for idx in indices[0]]
    return retrieved

def summarize_passages(passages: List[str], tokenizer, model, max_length: int = 50) -> str:
    if not passages:
        return ""
    text = " ".join(passages)
    inputs = tokenizer(f"Summarize: {text}", return_tensors="pt", max_length=max_length, truncation=True).to(model_device)
    with torch.no_grad():
        summary_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    del inputs, summary_ids
    clear_gpu_memory()
    return summary

def adacomp_rag_pipeline(prompts: List[str], dataset_name: str, batch_size: int = 1) -> List[Dict]:
    outputs = []
    retrieval_latencies = []
    contexts = data[dataset_name]["contexts"]
    index = faiss_indices[dataset_name]["index"]

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"🔄 AdaComp-RAG ({dataset_name})"):
            batch = prompts[i:i + batch_size]
            batch_idx = i // batch_size + 1
            try:
                t0 = time.time()
                log_vram_usage(f"Batch {batch_idx} start")

                batch_outputs = []
                for prompt in batch:
                    result = {"text": "", "mode": "", "context": ""}
                    mode = query_classifier(prompt, embedder)
                    result["mode"] = mode
                    if mode == "direct":
                        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True).to(model_device)
                        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0)
                        text = tokenizer.decode(gen[0], skip_special_tokens=True)
                        del inputs, gen
                    else:
                        t_retrieve = time.time()
                        passages = retrieve_passages(prompt, dataset_name, embedder, index, contexts, top_k)
                        retrieval_latencies.append(time.time() - t_retrieve)
                        summary = summarize_passages(passages, tokenizer, model)
                        result["context"] = summary
                        augmented_prompt = f"{prompt}\nContext: {summary}"
                        inputs = tokenizer(augmented_prompt, return_tensors="pt", max_length=max_length, truncation=True).to(model_device)
                        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0)
                        text = tokenizer.decode(gen[0], skip_special_tokens=True)
                        del inputs, gen
                    result["text"] = text
                    print(f"Generated output (Mode: {mode}): {text[:200]}...")  # Debug print
                    batch_outputs.append(result)
                    clear_gpu_memory()

                outputs.extend(batch_outputs)
                print(f"Batch {batch_idx} processed in {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                outputs.extend([{"text": "", "mode": "", "context": ""}] * len(batch))
                retrieval_latencies.append(0)
                clear_gpu_memory()
                continue

    return outputs, retrieval_latencies

# Evaluation
def extract_answer(text: str, dataset_name: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    if dataset_name == "gsm8k":
        patterns = [
            r"\*\*Final Answer: ([-+]?\d*\.?\d+)\*\*",
            r"\*\*Final Answer: (\d+)\*\*",
            r"(?:final answer|answer is)\s*[:=]?\s*([-+]?\d*\.?\d+)",
            r"(?:final answer|answer is)\s*[:=]?\s*(\d+)",
            r"\b([-+]?\d*\.?\d+)\s*(?:$|\D)",
            r"\b(\d+)\s*(?:$|\D)",
            r"[-+]?\d*\.?\d+\b"
        ]
    else:
        patterns = [
            r"(?:answer|option)\s*[:=]?\s*([A-D])",
            r"\b([A-D])\b"
        ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    print(f"Warning: No answer extracted from text: {text[:200]}...")
    return "0" if dataset_name == "gsm8k" else "A"

def calculate_memory_usage() -> float:
    return psutil.Process().memory_info().rss / (1024**3)

def run_evaluation(dataset_name: str, prompts: List[str], references: List[str], batch_size: int = 1) -> Dict:
    initial_memory = calculate_memory_usage()
    results = {
        "retrieval_latencies_ms": [],
        "f1_scores": [],
        "knowledge_retention_percent": [],
        "memory_usage_gb": [],
        "query_processing_time_ms": []
    }

    t0 = time.time()
    outputs, retrieval_latencies = adacomp_rag_pipeline(prompts, dataset_name, batch_size)
    t1 = time.time()

    batch_latency = (t1 - t0) / (len(prompts) / batch_size) if len(prompts) > 0 else 0
    query_processing_time_ms = batch_latency * 1000  # Convert seconds to milliseconds

    preds = [extract_answer(o["text"], dataset_name) for o in outputs]
    refs = references
    valid_pairs = [(p, r) for p, r in zip(preds, refs) if p != "0" and r]
    preds_numeric, refs_numeric = [], []
    for p, r in valid_pairs:
        try:
            preds_numeric.append(float(p))
            refs_numeric.append(float(r))
        except ValueError:
            continue

    try:
        if valid_pairs:
            binary_labels = [1 if abs(p - r) < 1e-6 else 0 for p, r in zip(preds_numeric, refs_numeric)]
            f1 = f1_score(binary_labels, [1]*len(binary_labels), average="macro")
        else:
            f1 = 0
    except Exception as e:
        print(f"Error computing F1: {e}")
        f1 = 0

    knowledge_retention_scores = []

    for prompt, output, ref in zip(prompts, outputs, refs):
        gen = output["text"]
        mode = output["mode"]
        context = output["context"]

        # Knowledge Retention: Exact Number Matching
        try:
            source = context if mode == "retrieve" and context else prompt
            retention = compute_knowledge_retention(source, gen)
            knowledge_retention_scores.append(retention)
            print(f"Knowledge Retention (Mode: {mode}): {retention:.2f}% for source: {source[:50]}...")
        except Exception as e:
            print(f"Debug: Error in knowledge retention: {e}")
            knowledge_retention_scores.append(0)

    final_memory = calculate_memory_usage()
    retrieval_latency_ms = np.mean([t * 1000 for t in retrieval_latencies]) if retrieval_latencies else 0  # Convert to ms

    results["retrieval_latencies_ms"].append(retrieval_latency_ms)
    results["f1_scores"].append(f1)
    results["knowledge_retention_percent"].append(np.mean(knowledge_retention_scores) if knowledge_retention_scores else 0)
    results["memory_usage_gb"].append(final_memory)
    results["query_processing_time_ms"].append(query_processing_time_ms)

    summary = {k: np.mean(v) if v else 0 for k, v in results.items()}

    print(f"\n===== AdaComp-RAG Evaluation Metrics ({dataset_name}) =====")
    print(f"Retrieval Latency (ms):      {summary['retrieval_latencies_ms']:.2f}")
    print(f"F1 Score:                   {summary['f1_scores']:.3f}")
    print(f"Knowledge Retention (%):    {summary['knowledge_retention_percent']:.2f}")
    print(f"Memory Usage (GB):          {summary['memory_usage_gb']:.2f}")
    print(f"Query Processing Time (ms): {summary['query_processing_time_ms']:.2f}")

    try:
        with open(f"adacomp_rag_{dataset_name}_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved to adacomp_rag_{dataset_name}_metrics.json")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    return summary

# Run evaluation for gsm8k
try:
    clear_gpu_memory()
    metrics = {}
    print(f"\nEvaluating on {dataset_name}...")
    metrics[dataset_name] = run_evaluation(dataset_name, data[dataset_name]["prompts"], data[dataset_name]["references"], batch_size)
    print("\n✅ AdaComp-RAG Evaluation Complete for GSM8K!")
except Exception as e:
    print(f"Error during evaluation: {e}")
    raise

# Clean up
for obj in ['model', 'tokenizer', 'embedder']:
    if obj in locals():
        del locals()[obj]
clear_gpu_memory()






#AdaComp on ARC dataset

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)

import subprocess
import sys
import torch
import numpy as np
import re
import gc
import psutil
import time
import json
from typing import List, Dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from accelerate import Accelerator

# Verify dependencies
try:
    import bitsandbytes
    from bitsandbytes import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    print(f"Bitsandbytes version: {bitsandbytes.__version__}")
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    if not cuda_version:
        print("Warning: CUDA not detected. Bitsandbytes may not work correctly.")
        BITSANDBYTES_AVAILABLE = False
except ImportError:
    print("Warning: bitsandbytes not installed. Falling back to float16.")
    BITSANDBYTES_AVAILABLE = False

import transformers
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Initialize Accelerator
accelerator = Accelerator()
print(f"Accelerator device: {accelerator.device}")

# Hugging Face login
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(hf_token)
    print("Successfully logged in to HuggingFace.")
except Exception as e:
    print(f"Error accessing HF_TOKEN: {e}")
    raise Exception("Authentication failed. Please set HF_TOKEN in Kaggle Secrets.")

# Set PyTorch environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuration
model_name = "google/gemma-7b"
dataset_name = "arc_challenge"
max_samples = 100
batch_size = 1
max_length = 128  # Increased to handle ARC prompts
max_new_tokens = 200  # Increased for complete answers
embedding_dim = 384
top_k = 3

# Use accelerator device
device = accelerator.device
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("Running on CPU")

# Dataset Loading and Preprocessing
def preprocess_arc(examples):
    prompts = []
    refs = []
    contexts = []  # Renamed from explanations for RAG consistency
    answer_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    try:
        for i in range(len(examples["question"])):
            question_stem = examples["question"][i]
            answer = answer_map.get(examples["answerKey"][i], examples["answerKey"][i])
            choices = examples["choices"][i]
            
            # Convert choices to list of dictionaries
            choice_list = [
                {"label": label, "text": text}
                for label, text in zip(choices["label"], choices["text"])
            ]
            
            formatted_choices = "\n".join([f"{c['label']}: {c['text']}" for c in choice_list])
            prompt = f"Question: {question_stem}\n{formatted_choices}\nAnswer:"
            context = f"{question_stem} " + " ".join([c['text'] for c in choice_list])
            
            prompts.append(prompt)
            refs.append(answer)
            contexts.append(context)
    except Exception as e:
        print(f"Error in preprocess_arc: {e}")
        raise

    return {"prompts": prompts, "references": refs, "contexts": contexts}

print("Loading dataset...")
try:
    datasets = {
        "arc_challenge": load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=42).select(range(max_samples))
    }
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise
data = {
    "arc_challenge": preprocess_arc(datasets["arc_challenge"])
}

# VRAM logging
def log_vram_usage(step: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"{step} - VRAM Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")

# Clear GPU memory
def clear_gpu_memory():
    for _ in range(3):  # Multiple GC runs
        gc.collect()
    if torch.cuda.is_available():
        for _ in range(2):  # Multiple cache clears
            torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        log_vram_usage("After GPU memory clear")

# Model & Tokenizer Loading
def load_model_and_tokenizer(model_name, quantize=True):
    log_vram_usage(f"Before loading {model_name}")
    clear_gpu_memory()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if BITSANDBYTES_AVAILABLE and quantize:
            print(f"Using 4-bit quantization for {model_name}...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            model.gradient_checkpointing_enable()
        else:
            print(f"Using float16 for {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        log_vram_usage(f"After loading {model_name}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        raise Exception("Failed to load LLM.")

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')  # Use CPU to save VRAM
print(f"Embedding model loaded on cpu")

# Load LLM
print(f"Loading LLM ({model_name})...")
model, tokenizer = load_model_and_tokenizer(model_name, quantize=True)
model, tokenizer = accelerator.prepare(model, tokenizer)
model_device = next(model.parameters()).device
print(f"Model is on device: {model_device}")

# Build FAISS index for contexts
def build_faiss_index(contexts: List[str], embedder, dim: int) -> faiss.IndexFlatL2:
    embeddings = embedder.encode(contexts, convert_to_numpy=True, show_progress_bar=True, device='cpu')
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

faiss_indices = {}
print(f"Building FAISS index for {dataset_name}...")
index, embeddings = build_faiss_index(data[dataset_name]["contexts"], embedder, embedding_dim)
faiss_indices[dataset_name] = {"index": index, "embeddings": embeddings}

# Knowledge Retention Function
def compute_knowledge_retention(source: str, generated: str) -> float:
    """Compute knowledge retention as the percentage of exact numerical matches."""
    if not generated or not source:
        print(f"Debug: Empty source or generated text (source: {source[:50]}..., generated: {generated[:50]}...)")
        return 0.0
    source_numbers = set(re.findall(r'[-+]?\d*\.?\d+', source))
    generated_numbers = set(re.findall(r'[-+]?\d*\.?\d+', generated))
    if not source_numbers:
        print(f"Debug: No numbers extracted from source: {source[:50]}...")
        return 0.0
    matches = len(source_numbers.intersection(generated_numbers))
    retention = matches / len(source_numbers)
    print(f"Debug: Source numbers: {source_numbers}, Generated numbers: {generated_numbers}, Matches: {matches}, Retention: {retention:.3f}")
    return retention * 100  # Convert to percentage

# AdaComp-RAG Functions
def query_classifier(query: str, embedder) -> str:
    tokens = len(query.split())
    has_numbers = bool(re.search(r'\d+', query))
    if tokens < 20 or has_numbers:
        return "direct"
    return "retrieve"

def retrieve_passages(query: str, dataset_name: str, embedder, index: faiss.IndexFlatL2, contexts: List[str], top_k: int = 3) -> List[str]:
    query_emb = embedder.encode([query], convert_to_numpy=True, device='cpu')
    distances, indices = index.search(query_emb, top_k)
    retrieved = [contexts[idx] for idx in indices[0]]
    return retrieved

def summarize_passages(passages: List[str], tokenizer, model, max_length: int = 50) -> str:
    if not passages:
        return ""
    text = " ".join(passages)
    inputs = tokenizer(f"Summarize: {text}", return_tensors="pt", max_length=max_length, truncation=True).to(model_device)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    del inputs, summary_ids
    clear_gpu_memory()
    return summary

def adacomp_rag_pipeline(prompts: List[str], dataset_name: str, batch_size: int = 1) -> List[Dict]:
    outputs = []
    retrieval_latencies = []
    contexts = data[dataset_name]["contexts"]
    index = faiss_indices[dataset_name]["index"]

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"🔄 AdaComp-RAG ({dataset_name})"):
            batch = prompts[i:i + batch_size]
            batch_idx = i // batch_size + 1
            try:
                t0 = time.time()
                log_vram_usage(f"Batch {batch_idx} start")

                batch_outputs = []
                for prompt in batch:
                    result = {"text": "", "mode": "", "context": ""}
                    mode = query_classifier(prompt, embedder)
                    result["mode"] = mode
                    if mode == "direct":
                        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True).to(model_device)
                        gen = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7
                        )
                        text = tokenizer.decode(gen[0], skip_special_tokens=True)
                        del inputs, gen
                    else:
                        t_retrieve = time.time()
                        passages = retrieve_passages(prompt, dataset_name, embedder, index, contexts, top_k)
                        retrieval_latencies.append(time.time() - t_retrieve)
                        summary = summarize_passages(passages, tokenizer, model)
                        result["context"] = summary
                        augmented_prompt = f"{prompt}\nContext: {summary}"
                        inputs = tokenizer(augmented_prompt, return_tensors="pt", max_length=max_length, truncation=True).to(model_device)
                        gen = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7
                        )
                        text = tokenizer.decode(gen[0], skip_special_tokens=True)
                        del inputs, gen
                    result["text"] = text
                    print(f"Generated output (Mode: {mode}): {text[:200]}...")  # Debug print
                    batch_outputs.append(result)
                    clear_gpu_memory()

                outputs.extend(batch_outputs)
                print(f"Batch {batch_idx} processed in {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                outputs.extend([{"text": "", "mode": "", "context": ""}] * len(batch))
                retrieval_latencies.append(0)
                clear_gpu_memory()
                continue

    return outputs, retrieval_latencies

# Evaluation
def extract_answer(text: str, dataset_name: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    if dataset_name == "arc_challenge":
        patterns = [
            r"(?:answer|option)\s*[:=]?\s*([A-D])",
            r"\b([A-D])\b"
        ]
    else:
        patterns = [
            r"\*\*Final Answer: ([-+]?\d*\.?\d+)\*\*",
            r"\*\*Final Answer: (\d+)\*\*",
            r"(?:final answer|answer is)\s*[:=]?\s*([-+]?\d*\.?\d+)",
            r"(?:final answer|answer is)\s*[:=]?\s*(\d+)",
            r"\b([-+]?\d*\.?\d+)\s*(?:$|\D)",
            r"\b(\d+)\s*(?:$|\D)",
            r"([-+]?\d*\.?\d+)\b"  # Fixed: Added capturing group
        ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            print(f"Debug: Matched pattern '{pattern}' in text: {text[:100]}... -> {match.group(1)}")
            return match.group(1).strip()
    print(f"Warning: No answer extracted from text: {text[:200]}...")
    return "A" if dataset_name == "arc_challenge" else "0"

def calculate_memory_usage() -> float:
    return psutil.Process().memory_info().rss / (1024**3)

def run_evaluation(dataset_name: str, prompts: List[str], references: List[str], batch_size: int = 1) -> Dict:
    initial_memory = calculate_memory_usage()
    results = {
        "retrieval_latencies_ms": [],
        "f1_scores": [],
        "knowledge_retention_percent": [],
        "memory_usage_gb": [],
        "query_processing_time_ms": []
    }

    t0 = time.time()
    outputs, retrieval_latencies = adacomp_rag_pipeline(prompts, dataset_name, batch_size)
    t1 = time.time()

    batch_latency = (t1 - t0) / (len(prompts) / batch_size) if len(prompts) > 0 else 0
    query_processing_time_ms = batch_latency * 1000  # Convert seconds to milliseconds

    preds = [extract_answer(o["text"], dataset_name) for o in outputs]
    refs = references
    valid_pairs = [(p, r) for p, r in zip(preds, refs) if p != "A" and r]
    preds_numeric, refs_numeric = [], []
    for p, r in valid_pairs:
        try:
            # Map A, B, C, D to 0, 1, 2, 3 for numerical comparison
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            preds_numeric.append(mapping[p])
            refs_numeric.append(mapping[r])
        except (ValueError, KeyError):
            continue

    try:
        if valid_pairs:
            binary_labels = [1 if p == r else 0 for p, r in zip(preds_numeric, refs_numeric)]
            f1 = f1_score(binary_labels, [1]*len(binary_labels), average="macro")
        else:
            f1 = 0
    except Exception as e:
        print(f"Error computing F1: {e}")
        f1 = 0

    knowledge_retention_scores = []

    for prompt, output, ref in zip(prompts, outputs, refs):
        gen = output["text"]
        mode = output["mode"]
        context = output["context"]

        # Knowledge Retention: Exact Number Matching
        try:
            source = context if mode == "retrieve" and context else prompt
            retention = compute_knowledge_retention(source, gen)
            knowledge_retention_scores.append(retention)
            print(f"Knowledge Retention (Mode: {mode}): {retention:.2f}% for source: {source[:50]}...")
        except Exception as e:
            print(f"Debug: Error in knowledge retention: {e}")
            knowledge_retention_scores.append(0)

    final_memory = calculate_memory_usage()
    retrieval_latency_ms = np.mean([t * 1000 for t in retrieval_latencies]) if retrieval_latencies else 0  # Convert to ms

    results["retrieval_latencies_ms"].append(retrieval_latency_ms)
    results["f1_scores"].append(f1)
    results["knowledge_retention_percent"].append(np.mean(knowledge_retention_scores) if knowledge_retention_scores else 0)
    results["memory_usage_gb"].append(final_memory)
    results["query_processing_time_ms"].append(query_processing_time_ms)

    summary = {k: np.mean(v) if v else 0 for k, v in results.items()}

    print(f"\n===== AdaComp-RAG Evaluation Metrics ({dataset_name}) =====")
    print(f"Retrieval Latency (ms):      {summary['retrieval_latencies_ms']:.2f}")
    print(f"F1 Score:                   {summary['f1_scores']:.3f}")
    print(f"Knowledge Retention (%):    {summary['knowledge_retention_percent']:.2f}")
    print(f"Memory Usage (GB):          {summary['memory_usage_gb']:.2f}")
    print(f"Query Processing Time (ms): {summary['query_processing_time_ms']:.2f}")

    try:
        with open(f"adacomp_rag_{dataset_name}_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved to adacomp_rag_{dataset_name}_metrics.json")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    return summary

# Run evaluation for arc_challenge
try:
    clear_gpu_memory()
    metrics = {}
    print(f"\nEvaluating on {dataset_name}...")
    metrics[dataset_name] = run_evaluation(dataset_name, data[dataset_name]["prompts"], data[dataset_name]["references"], batch_size)
    print("\n AdaComp-RAG Evaluation Complete for ARC-Challenge!")
except Exception as e:
    print(f"Error during evaluation: {e}")
    raise

# Clean up
for obj in ['model', 'tokenizer', 'embedder']:
    if obj in locals():
        del locals()[obj]
clear_gpu_memory()