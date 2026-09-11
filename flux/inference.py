import os
import json
import torch
import multiprocessing
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from tqdm import tqdm
import argparse

# Global configuration (will be set by init_worker)
MODEL_PATH = None
LORA_PATH = None
OUTPUT_DIR = None

pipe = None
initialized = False

def init_worker(gpu_queue, model_path, lora_path, output_dir):
    global pipe, initialized, MODEL_PATH, LORA_PATH, OUTPUT_DIR
    if initialized:
        return

    # Set global configuration for this worker
    MODEL_PATH = model_path
    LORA_PATH = lora_path
    OUTPUT_DIR = output_dir

    try:
        gpu_id = gpu_queue.get(timeout=5)
    except Exception as e:
        print(f"Failed to get GPU ID: {e}")
        gpu_id = 0

    device = f"cuda:{gpu_id}"
    print(f"Initializing worker on {device}")

    pipe = FluxKontextPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(LORA_PATH, weight_name="pytorch_lora_weights.safetensors")
    pipe.to(device)
    initialized = True

def process_batch(items):
    global pipe, OUTPUT_DIR
    if not initialized:
        raise RuntimeError("Worker not initialized")

    images = []
    prompts = []
    save_paths = []
    orig_sizes = []
    target_size = (1024, 1024)

    for item in items:
        src_path = item.get("source")
        target_path = item.get("target") or src_path
        filename = os.path.basename(target_path)
        text = item.get("text", "")
        save_path = os.path.join(OUTPUT_DIR, filename)

        # Load and preprocess image
        try:
            img_in = load_image(src_path)
        except Exception as e:
            print(f"Error loading image {src_path}: {e}")
            continue

        orig_w, orig_h = img_in.size
        img_resized = img_in.resize(target_size)

        images.append(img_resized)
        prompts.append(text)
        save_paths.append(save_path)
        orig_sizes.append((orig_w, orig_h))

    if not images:
        return []

    # Inference
    try:
        with torch.no_grad():
            outputs = pipe(
                image=images,
                prompt=prompts,
                guidance_scale=3.5,
                generator=torch.Generator().manual_seed(123456),

            ).images

        # Post-process and save
        saved_files = []
        for output, save_path, orig_size in zip(outputs, save_paths, orig_sizes):
            output = output.resize(orig_size)
            output.save(save_path)
            saved_files.append(save_path)

        return saved_files
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    items = []
    with open(args.testset_path, "r") as f:
        for line in tqdm(f, desc="Scanning dataset"):
            item = json.loads(line.strip())
            src_rel = item.get("source")
            filename = os.path.basename(item.get("target") or src_rel)

            # Check if already processed
            if os.path.exists(os.path.join(args.output_dir, filename)):
                continue

            items.append(item)

    # Chunk items into batches
    batches = [items[i:i + args.batch_size] for i in range(0, len(items), args.batch_size)]

    print(f"Total samples: {len(items)}, Total batches: {len(batches)}")

    # Setup GPU distribution
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    if num_gpus == 0:
        print("No GPUs found, exiting.")
        return

    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    gpu_queue = manager.Queue()

    for i in range(args.workers):
        gpu_queue.put(i % num_gpus)

    with ctx.Pool(processes=args.workers, initializer=init_worker, initargs=(gpu_queue, args.model_path, args.lora_path, args.output_dir)) as pool:
        for _ in tqdm(pool.imap_unordered(process_batch, batches), total=len(batches), desc="Processing batches"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, default="/path/to/FLUX.1-Kontext-dev")
    parser.add_argument("--lora_path", type=str, default="/path/to/flux_cft/checkpoint-4000")
    parser.add_argument("--output_dir", type=str, default="/path/to/flux_results")
    parser.add_argument("--testset_path", type=str, default="/path/to/testset.jsonl")

    args = parser.parse_args()
    main(args)
