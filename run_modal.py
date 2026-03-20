"""
Parameter Golf — Run on Modal with 8xH100.
Usage:
    modal run run_modal.py
"""
import modal
import os

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
    )
)

app = modal.App("parameter-golf", image=image)

# Create a volume for the dataset (persistent, download once)
dataset_vol = modal.Volume.from_name("pg-dataset", create_if_missing=True)

@app.function(
    gpu=modal.gpu.H100(count=8),
    timeout=1800,  # 30 min max (training + eval + overhead)
    volumes={"/data": dataset_vol},
)
def train(seed: int = 1337):
    import subprocess
    import shutil

    # Download dataset if not already cached in the volume
    dataset_dir = "/data/datasets/fineweb10B_sp1024"
    tokenizer_path = "/data/tokenizers/fineweb_1024_bpe.model"

    if not os.path.exists(tokenizer_path):
        print("Downloading dataset (first run only)...")
        # Copy the download script
        subprocess.run(
            ["python3", "/root/parameter-golf/data/cached_challenge_fineweb.py",
             "--variant", "sp1024", "--train-shards", "10"],
            cwd="/root/parameter-golf",
            check=True,
            env={**os.environ, "HF_HOME": "/data/.cache/huggingface"},
        )
        # Move downloaded data to volume
        src = "/root/parameter-golf/data/datasets/fineweb10B_sp1024"
        if os.path.exists(src) and not os.path.exists(dataset_dir):
            shutil.copytree(src, dataset_dir)
        src_tok = "/root/parameter-golf/data/tokenizers"
        dst_tok = "/data/tokenizers"
        if os.path.exists(src_tok) and not os.path.exists(dst_tok):
            shutil.copytree(src_tok, dst_tok)
        dataset_vol.commit()

    # Upload and run the training script
    script_path = "/root/parameter-golf/my_train.py"

    # Clone and setup
    if not os.path.exists("/root/parameter-golf"):
        subprocess.run(
            ["git", "clone", "https://github.com/rajkotraja/parameter-golf-SR.git",
             "/root/parameter-golf"],
            check=True,
        )

    env = {
        **os.environ,
        "RUN_ID": f"modal_seed{seed}",
        "SEED": str(seed),
        "NUM_LAYERS": "10",
        "WARMDOWN_ITERS": "2500",
        "TIED_EMBED_LR": "0.10",
        "EVAL_STRIDE": "64",
        "EVAL_SEQ_LEN": "1024",
        "MAX_WALLCLOCK_SECONDS": "600",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "200",
        "DATA_PATH": dataset_dir,
        "TOKENIZER_PATH": tokenizer_path,
    }

    print(f"Starting training with seed={seed} on 8xH100...")
    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=8", script_path],
        env=env,
        cwd="/root/parameter-golf",
        capture_output=False,
    )

    # Copy logs to volume for persistence
    log_file = f"/root/parameter-golf/logs/modal_seed{seed}.txt"
    if os.path.exists(log_file):
        shutil.copy(log_file, f"/data/train_seed{seed}.log")
        dataset_vol.commit()
        print(f"Log saved to volume: /data/train_seed{seed}.log")

    return result.returncode


@app.local_entrypoint()
def main(seed: int = 1337):
    print(f"Launching Parameter Golf training on Modal (seed={seed})...")
    print("This will take ~15 minutes and cost ~$3-5")
    returncode = train.remote(seed=seed)
    if returncode == 0:
        print("Training completed successfully!")
    else:
        print(f"Training failed with exit code {returncode}")
