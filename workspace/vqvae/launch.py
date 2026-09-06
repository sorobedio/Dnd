"""Wait for physical GPU 4 before importing PyTorch or initializing CUDA."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


def free_memory_mib():
    result = subprocess.run(
        ["nvidia-smi", "-i", "4", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def wait_for_memory(minimum_mib, interval):
    if minimum_mib < 1 or interval <= 0:
        raise ValueError("Memory threshold and polling interval must be positive")
    while True:
        free = free_memory_mib()
        if free >= minimum_mib:
            print(f"GPU 4 has {free} MiB free; starting VQ-VAE training.", flush=True)
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"{timestamp} Waiting for GPU 4: {free} MiB free; "
              f"need at least {minimum_mib} MiB. Existing jobs are left running.", flush=True)
        time.sleep(interval)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    reconstruction = '--reconstruction' in argv
    argv = [arg for arg in argv if arg != '--reconstruction']
    command = ([sys.executable, '-m', 'workspace.vqvae.train_reconstruction', '--encode-after', *argv]
               if reconstruction else [sys.executable, "-m", "workspace.vqvae.run", "train", "--encode-after", *argv])
    if "--help" in argv or "-h" in argv:
        os.execv(sys.executable, command)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] /
        ('outputs/vqvae_train_arc_c_refined' if reconstruction else 'outputs/vqvae_arc_c')))
    parser.add_argument("--resume")
    args, _ = parser.parse_known_args(argv)
    output = Path(args.output_dir)
    if (output / "last.pt").exists() and not args.resume:
        raise SystemExit("A checkpoint exists; pass --resume or choose a new --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    # Keep this lock across exec so repeated launch attempts cannot queue duplicate runs.
    lock = os.open(output / ".launcher.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock)
        raise SystemExit(f"A VQ-VAE run is already queued or running for {output}.")
    os.set_inheritable(lock, True)
    os.ftruncate(lock, 0)
    os.write(lock, f"pid={os.getpid()} gpu=4\n".encode())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    if args.device.startswith("cuda"):
        wait_for_memory(int(os.environ.get("VQVAE_MIN_FREE_MIB", "16384")),
                        float(os.environ.get("VQVAE_POLL_SECONDS", "30")))
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
