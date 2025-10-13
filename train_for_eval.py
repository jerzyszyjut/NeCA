import torch
import argparse
import multiprocessing as mp
from pathlib import Path

from src.config.configloading import load_config
from src.eval_trainer import EvalTrainer


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/neca/", help="data path")
    return parser


def discover_config_files(data_dir):
    config_files = []
    data_path = Path(data_dir)

    for config_file in data_path.glob("**/config.yaml"):
        config_files.append(str(config_file))

    return config_files


def train_worker(gpu_id, config_files):
    device = f"cuda:{gpu_id}"

    # Set CUDA device for this worker
    torch.cuda.set_device(gpu_id)

    print(
        f"GPU {gpu_id}: Starting sequential processing of {len(config_files)} configurations"
    )

    for i, config_file in enumerate(config_files, 1):
        print(f"GPU {gpu_id}: Starting training {i}/{len(config_files)}: {config_file}")

        trainer = None
        cfg = None

        try:
            # Clear CUDA cache before each training
            torch.cuda.empty_cache()

            cfg = load_config(config_file)
            trainer = EvalTrainer(cfg, device)
            trainer.start()

            print(
                f"GPU {gpu_id}: ✓ Completed training {i}/{len(config_files)}: {config_file}"
            )
        except Exception as e:
            print(
                f"GPU {gpu_id}: ✗ Error training {i}/{len(config_files)} {config_file}: {e}"
            )
        finally:
            # Clean up resources
            try:
                if trainer is not None:
                    del trainer
                if cfg is not None:
                    del cfg
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"GPU {gpu_id}: Finished all {len(config_files)} configurations")


def distribute_configs(config_files, num_gpus):
    distributed_configs = [[] for _ in range(num_gpus)]

    for i, config_file in enumerate(config_files):
        gpu_id = i % num_gpus
        distributed_configs[gpu_id].append(config_file)

    return distributed_configs


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    parser = config_parser()
    args = parser.parse_args()

    config_files = discover_config_files(args.data_dir)

    if not config_files:
        print(f"No config.yaml files found in {args.data_dir}")
        return

    print(f"Found {len(config_files)} config files:")
    for config_file in config_files:
        print(f"  - {config_file}")

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available")

    distributed_configs = distribute_configs(config_files, num_gpus)

    processes = []
    for gpu_id in range(num_gpus):
        if distributed_configs[gpu_id]:
            p = mp.Process(
                target=train_worker, args=(gpu_id, distributed_configs[gpu_id])
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
