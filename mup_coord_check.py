#!/usr/bin/env python
"""
MuP Verification Plots for Megatron-LM

Generates publication-quality plots that demonstrate MuP correctness:
1. Coordinate Check: Activation stability across widths
2. LR Transfer: Optimal LR is width-invariant with MuP
3. Loss Curves: Training dynamics align across widths

Setup (download test data):
    # Download official Megatron-LM test datasets
    mkdir -p assets && cd assets
    wget https://github.com/NVIDIA/Megatron-LM/releases/download/v2.5/datasets.zip
    wget https://github.com/NVIDIA/Megatron-LM/releases/download/v2.5/tokenizers.zip
    unzip datasets.zip && unzip tokenizers.zip
    cd ..

Run on A100 80GB:
    # With real data (recommended for NVIDIA review)
    python mup_plots.py --output-dir ./mup_plots --data-dir ./assets

    # With mock data (faster, no setup required)
    python mup_plots.py --output-dir ./mup_plots --use-mock-data

Expected results with correct MuP:
- Coordinate check: Flat lines for MuP, growing lines for standard
- LR transfer: Same optimal LR across widths for MuP
- Loss curves: Overlapping curves for MuP at different widths
"""

import os
import sys
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set minimal distributed env for single GPU
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '29500')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')


def init_distributed():
    """Initialize minimal distributed setup for single GPU."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            world_size=1,
            rank=0,
        )

    from megatron.core import parallel_state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(42)


def cleanup():
    """Cleanup distributed."""
    from megatron.core import parallel_state
    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class DataProvider:
    """Provides training data from either real datasets or mock data."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        use_mock: bool = False,
        vocab_size: int = 50257,  # GPT-2 vocab size
        seq_len: int = 128,
        batch_size: int = 4,
    ):
        self.data_dir = data_dir
        self.use_mock = use_mock
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._dataset = None
        self._dataloader = None
        self._iterator = None

        if use_mock:
            self._setup_mock_data()
        else:
            # Use real data - will auto-download if needed
            if not data_dir:
                self.data_dir = "./assets"  # Default location
            self._setup_real_data()

    def _download_test_data(self, data_path: Path):
        """Download and preprocess a real dataset (WikiText-103 subset)."""
        import subprocess

        data_path.mkdir(parents=True, exist_ok=True)

        # Download GPT-2 tokenizer files
        tokenizer_files = {
            "gpt2-vocab.json": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
            "gpt2-merges.txt": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        }

        import requests
        for filename, url in tokenizer_files.items():
            filepath = data_path / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                response = requests.get(url)
                response.raise_for_status()
                with open(filepath, 'w') as f:
                    f.write(response.text)

        # Check if preprocessed data exists
        bin_files = list(data_path.glob("*_text_document.bin"))
        if bin_files:
            print("Preprocessed data already exists.")
            return

        # Download WikiText-103 raw data using HuggingFace datasets
        print("Downloading WikiText-103 dataset...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:50000]")

            # Save as JSONL for Megatron preprocessing
            jsonl_path = data_path / "wikitext_train.jsonl"
            print(f"Saving to {jsonl_path}...")
            import json
            with open(jsonl_path, 'w') as f:
                for item in dataset:
                    text = item['text'].strip()
                    if text:  # Skip empty lines
                        f.write(json.dumps({"text": text}) + "\n")

            # Preprocess with Megatron using HuggingFace tokenizer
            print("Preprocessing with Megatron...")
            preprocess_cmd = [
                sys.executable, "tools/preprocess_data.py",
                "--input", str(jsonl_path),
                "--output-prefix", str(data_path / "wikitext"),
                "--tokenizer-type", "HuggingFaceTokenizer",
                "--tokenizer-model", "gpt2",
                "--append-eod",
                "--workers", "4",
            ]
            result = subprocess.run(preprocess_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                if "AutoTokenizer" in result.stderr or "transformers" in result.stderr.lower():
                    raise RuntimeError(
                        "Preprocessing requires 'transformers' package. "
                        "Run: pip install transformers"
                    )
                raise RuntimeError(f"Preprocessing failed: {result.stderr}")
            print("Preprocessing complete.")

        except ImportError as e:
            if "datasets" in str(e):
                print("ERROR: 'datasets' package not installed. Run: pip install datasets")
            raise

    def _setup_real_data(self):
        """Setup real data from Megatron dataset files."""
        from megatron.core.datasets.utils import compile_helpers

        # Compile helpers
        compile_helpers()

        data_path = Path(self.data_dir)

        # Check for required files, download if missing
        tokenizer_path = data_path / "gpt2-vocab.json"
        merge_path = data_path / "gpt2-merges.txt"
        bin_files = list(data_path.glob("**/*_text_document.bin"))

        if not bin_files or not tokenizer_path.exists() or not merge_path.exists():
            print("Required data files not found. Downloading...")
            self._download_test_data(data_path)
            # Re-check after download
            tokenizer_path = data_path / "gpt2-vocab.json"
            merge_path = data_path / "gpt2-merges.txt"
            bin_files = list(data_path.glob("**/*_text_document.bin"))

        if not bin_files:
            raise RuntimeError(f"No preprocessed .bin files found in {data_path}")
        if not tokenizer_path.exists():
            raise RuntimeError(f"Tokenizer vocab file not found: {tokenizer_path}")
        if not merge_path.exists():
            raise RuntimeError(f"Tokenizer merge file not found: {merge_path}")

        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, GPTDataset
        from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
        from megatron.training.tokenizer.tokenizer import _GPT2BPETokenizer

        # Setup tokenizer
        tokenizer = _GPT2BPETokenizer(
            vocab_file=str(tokenizer_path),
            merge_file=str(merge_path),
        )
        self.vocab_size = tokenizer.vocab_size

        # Use the first found dataset - get the full prefix including _text_document
        data_prefix = str(bin_files[0]).replace(".bin", "")
        print(f"Using dataset: {data_prefix}")
        print(f"Vocab size: {self.vocab_size}")

        # Create cache directory
        cache_path = data_path / "cache"
        cache_path.mkdir(exist_ok=True)

        config = GPTDatasetConfig(
            random_seed=42,
            sequence_length=self.seq_len,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            tokenizer=tokenizer,
            path_to_cache=str(cache_path),
            blend=([data_prefix], None),  # (paths_list, weights_or_None)
            split="99,1,0",  # 99% train, 1% valid, 0% test
        )

        datasets = BlendedMegatronDatasetBuilder(
            GPTDataset,
            [None, None, None],  # Use all available data
            lambda: True,
            config,
        ).build()

        self._dataset = datasets[0]
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )
        self.use_mock = False
        print(f"Dataset size: {len(self._dataset)} samples")

    def _setup_mock_data(self):
        """Setup simple mock data generator (no Megatron dataset dependencies)."""
        # Use a smaller vocab for mock data to avoid issues
        self.vocab_size = 1024  # Small vocab for testing
        self.use_mock = True
        self._dataset = None
        self._dataloader = None
        print(f"Using simple mock data generator (vocab_size={self.vocab_size})")

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get a batch of data."""
        if self.use_mock:
            # Generate random data for mock mode
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device='cuda')
            position_ids = torch.arange(self.seq_len, device='cuda').unsqueeze(0).expand(self.batch_size, -1)
            labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device='cuda')
            loss_mask = torch.ones(self.batch_size, self.seq_len, device='cuda')
            return {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'labels': labels,
                'loss_mask': loss_mask,
            }

        # Real data mode
        if self._iterator is None:
            self._iterator = iter(self._dataloader)

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            batch = next(self._iterator)

        return {
            'input_ids': batch['tokens'].cuda(),
            'position_ids': batch['position_ids'].cuda(),
            'labels': batch['labels'].cuda(),
            'loss_mask': batch['loss_mask'].cuda(),
        }

    def reset(self):
        """Reset the iterator for reproducible experiments."""
        self._iterator = None


class CharLevelDataset(torch.utils.data.Dataset):
    """Simple character-level dataset for enwik8/text8."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = self.data[start:end]
        labels = self.data[start + 1:end + 1]
        position_ids = torch.arange(self.seq_len)
        loss_mask = torch.ones(self.seq_len)
        return {
            'tokens': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
        }


class CharLevelDataProvider:
    """Provides character-level training data from enwik8 or text8."""

    DATASETS = {
        'enwik8': {
            'url': 'http://mattmahoney.net/dc/enwik8.zip',
            'filename': 'enwik8',
            'vocab_size': 256,  # byte-level
        },
        'text8': {
            'url': 'http://mattmahoney.net/dc/text8.zip',
            'filename': 'text8',
            'vocab_size': 27,  # a-z + space
        },
    }

    def __init__(
        self,
        dataset_name: str = 'enwik8',
        data_dir: str = './assets',
        seq_len: int = 128,
        batch_size: int = 4,
    ):
        assert dataset_name in self.DATASETS, f"Unknown dataset: {dataset_name}. Choose from {list(self.DATASETS.keys())}"

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._iterator = None

        # Download and load data
        self._download_if_needed()
        self._load_data()

    def _download_if_needed(self):
        """Download dataset if not present."""
        import requests
        import zipfile

        self.data_dir.mkdir(parents=True, exist_ok=True)
        info = self.DATASETS[self.dataset_name]
        filepath = self.data_dir / info['filename']

        if filepath.exists():
            print(f"Dataset {self.dataset_name} already exists at {filepath}")
            return

        zip_path = self.data_dir / f"{info['filename']}.zip"
        if not zip_path.exists():
            print(f"Downloading {self.dataset_name} from {info['url']}...")
            response = requests.get(info['url'], stream=True)
            response.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {zip_path}")

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(self.data_dir)
        print(f"Extracted to {self.data_dir}")

    def _load_data(self):
        """Load and tokenize the dataset."""
        info = self.DATASETS[self.dataset_name]
        filepath = self.data_dir / info['filename']

        print(f"Loading {self.dataset_name} from {filepath}...")
        with open(filepath, 'rb') as f:
            raw_data = f.read()

        if self.dataset_name == 'text8':
            # text8 is already clean: lowercase a-z and space
            # Map to 0-26 (space=0, a=1, ..., z=26)
            char_to_idx = {' ': 0}
            for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
                char_to_idx[c] = i + 1
            data = torch.tensor([char_to_idx.get(chr(b), 0) for b in raw_data], dtype=torch.long)
            self.vocab_size = 27
            self.idx_to_char = {v: k for k, v in char_to_idx.items()}
        else:
            # enwik8: byte-level (0-255)
            data = torch.tensor(list(raw_data), dtype=torch.long)
            self.vocab_size = 256
            self.idx_to_char = {i: chr(i) if 32 <= i < 127 else f'<{i}>' for i in range(256)}

        # Split: 90M train, 5M valid, 5M test (standard split)
        train_size = 90_000_000
        valid_size = 5_000_000

        if len(data) >= train_size + valid_size:
            train_data = data[:train_size]
        else:
            # For smaller datasets, use 90% for training
            train_size = int(len(data) * 0.9)
            train_data = data[:train_size]

        print(f"Loaded {len(data):,} characters, using {len(train_data):,} for training")
        print(f"Vocab size: {self.vocab_size}")

        # Create dataset and dataloader
        self._dataset = CharLevelDataset(train_data, self.seq_len)
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get a batch of data."""
        if self._iterator is None:
            self._iterator = iter(self._dataloader)

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            batch = next(self._iterator)

        return {
            'input_ids': batch['tokens'].cuda(),
            'position_ids': batch['position_ids'].cuda(),
            'labels': batch['labels'].cuda(),
            'loss_mask': batch['loss_mask'].cuda(),
        }

    def reset(self):
        """Reset the iterator for reproducible experiments."""
        self._iterator = None


# Global data provider (initialized once)
_data_provider: Optional[DataProvider] = None


def get_data_provider(
    data_dir: Optional[str] = None,
    use_mock: bool = False,
    vocab_size: int = 50257,
    seq_len: int = 128,
    batch_size: int = 4,
) -> DataProvider:
    """Get or create the global data provider."""
    global _data_provider
    if _data_provider is None:
        _data_provider = DataProvider(
            data_dir=data_dir,
            use_mock=use_mock,
            vocab_size=vocab_size,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    return _data_provider


def create_gpt_model(hidden_size, num_layers, num_heads, use_mup, base_hidden_size, vocab_size=50257):
    """Create a minimal GPTModel for testing."""
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    config = TransformerConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        use_mup=use_mup,
        mup_base_hidden_size=base_hidden_size if use_mup else None,
        use_cpu_initialization=False,
        perform_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=512,
        pre_process=True,
        post_process=True,
    ).cuda()

    return model, config


def collect_activation_stats(model, config, batch_size=4, seq_len=128) -> Dict[str, float]:
    """Collect activation statistics from a forward pass."""
    vocab_size = 256

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)

    activation_stats = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out is not None and hasattr(out, 'float'):
                activation_stats[name] = {
                    'mean': out.float().mean().item(),
                    'std': out.float().std().item(),
                    'abs_mean': out.float().abs().mean().item(),
                }
        return hook

    # Register hooks on key layers
    for name, module in model.named_modules():
        if 'layernorm' in name.lower() or 'layer_norm' in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))
        elif 'attention' in name.lower() and 'output' not in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))
        elif 'mlp' in name.lower() and hasattr(module, 'weight'):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
        )

    activation_stats['logits'] = {
        'mean': logits.float().mean().item(),
        'std': logits.float().std().item(),
        'abs_mean': logits.float().abs().mean().item(),
    }

    for hook in hooks:
        hook.remove()

    return activation_stats


def collect_activation_stats_per_step(
    model,
    config,
    data_provider,
    num_steps: int = 10,
    lr: float = 1e-3,
) -> List[Dict[str, any]]:
    """Collect activation stats at each training step.

    This implements Microsoft's coord_check approach - tracking activation
    statistics throughout training to verify MuP correctness.

    Args:
        model: The GPT model to train
        config: TransformerConfig (needed for MuP LR scaling)
        data_provider: DataProvider for training batches
        num_steps: Number of training steps
        lr: Learning rate for optimizer

    Returns:
        List of dicts, one per step, containing activation statistics
        for each tracked layer.
    """
    from megatron.core.optimizer import get_mup_config_overrides
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    all_stats = []
    hooks = []
    current_activations = {}

    def make_hook(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out is not None and hasattr(out, 'float'):
                # Compute L1 norm (abs mean) and std
                out_float = out.float()
                current_activations[name] = {
                    'l1': out_float.abs().mean().item(),
                    'std': out_float.std().item(),
                }
        return hook

    # Register hooks on attention submodules and MLP
    tracked_patterns = [
        ('query', ['query', 'linear_q']),
        ('key', ['key', 'linear_k']),
        ('value', ['value', 'linear_v']),
        ('qkv', ['linear_qkv']),  # Combined QKV projection
        ('attn_score', ['core_attention']),  # Attention scores
        ('mlp_fc1', ['linear_fc1']),  # MLP first layer
        ('mlp_fc2', ['linear_fc2']),  # MLP second layer
    ]

    for name, module in model.named_modules():
        name_lower = name.lower()
        for pattern_name, patterns in tracked_patterns:
            if any(p in name_lower for p in patterns):
                # Use a short name for the layer (e.g., "layer0.query")
                # Extract layer number if present
                parts = name.split('.')
                layer_num = None
                for part in parts:
                    if part.isdigit():
                        layer_num = int(part)
                        break
                if layer_num is not None:
                    short_name = f"layer{layer_num}.{pattern_name}"
                else:
                    short_name = pattern_name
                hooks.append(module.register_forward_hook(make_hook(short_name)))
                break

    # Setup optimizer with MuP LR scaling if applicable
    use_mup = config is not None and hasattr(config, 'use_mup') and config.use_mup
    if use_mup and hasattr(config, 'mup_width_mult') and config.mup_width_mult != 1.0:
        opt_config = OptimizerConfig(lr=lr, min_lr=lr / 10)
        mup_overrides = get_mup_config_overrides(
            opt_config,
            mup_width_mult=config.mup_width_mult,
            optimizer_type='adam'
        )

        params_with_scaled_lr = []
        params_with_base_lr = []
        scaled_lr = lr

        for param_key, override in mup_overrides.items():
            scaled_lr = override.get('max_lr', lr)
            predicate_fn = param_key.with_name_predicate.fn

            for name, param in model.named_parameters():
                if predicate_fn(param, name):
                    params_with_scaled_lr.append(param)
                else:
                    params_with_base_lr.append(param)
            break

        param_groups = []
        if params_with_scaled_lr:
            param_groups.append({'params': params_with_scaled_lr, 'lr': scaled_lr})
        if params_with_base_lr:
            param_groups.append({'params': params_with_base_lr, 'lr': lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()

    vocab_size = data_provider.vocab_size

    for step in range(num_steps):
        batch = data_provider.get_batch()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(
                input_ids=batch['input_ids'],
                position_ids=batch['position_ids'],
                attention_mask=None,
            )
            # Compute loss
            loss_flat = F.cross_entropy(
                logits.view(-1, vocab_size).float(),
                batch['labels'].view(-1),
                reduction='none',
            )
            loss_mask_flat = batch['loss_mask'].view(-1)
            loss = (loss_flat * loss_mask_flat).sum() / loss_mask_flat.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Record stats for this step
        step_record = {
            'step': step,
            'loss': loss.item(),
            'logits_l1': logits.float().abs().mean().item(),
            'logits_std': logits.float().std().item(),
        }
        # Add per-layer activations
        step_record['layers'] = dict(current_activations)
        all_stats.append(step_record)
        current_activations = {}

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

    return all_stats


def collect_delta_from_init_stats(
    model,
    config,
    data_provider,
    num_steps: int = 5,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:
    """Collect std(x_t - x_0) statistics - change from initialization.

    This implements the exact metric from the MuP paper Figure 8:
    tracking how much activations change from their initial values
    as training progresses.

    Args:
        model: The GPT model to train
        config: TransformerConfig (needed for MuP LR scaling)
        data_provider: DataProvider for training batches
        num_steps: Number of training steps (t values)
        lr: Learning rate for optimizer

    Returns:
        Dict mapping metric names to list of std(x_t - x_0) values per step.
        Keys: 'logits', 'attn_logits', 'word_embedding'
    """
    from megatron.core.optimizer import get_mup_config_overrides
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    # Storage for initial activations and per-step deltas
    initial_activations = {}  # name -> tensor
    current_activations = {}  # name -> tensor
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out is not None and hasattr(out, 'detach'):
                current_activations[name] = out.detach().clone().float()
        return hook

    # Register hooks for the three metrics we want to track
    # 1. logits - output of the model
    # 2. attn_logits - attention scores (QK^T)
    # 3. word_embedding - embedding layer output

    for name, module in model.named_modules():
        name_lower = name.lower()
        # Word embedding
        if 'embedding' in name_lower and 'word' in name_lower:
            hooks.append(module.register_forward_hook(make_hook('word_embedding')))
        # Attention core (for attention logits)
        elif 'core_attention' in name_lower or 'self_attention' in name_lower:
            hooks.append(module.register_forward_hook(make_hook('attn_logits')))

    # Setup optimizer with MuP LR scaling
    use_mup = config is not None and hasattr(config, 'use_mup') and config.use_mup
    if use_mup and hasattr(config, 'mup_width_mult') and config.mup_width_mult != 1.0:
        opt_config = OptimizerConfig(lr=lr, min_lr=lr / 10)
        mup_overrides = get_mup_config_overrides(
            opt_config,
            mup_width_mult=config.mup_width_mult,
            optimizer_type='adam'
        )

        params_with_scaled_lr = []
        params_with_base_lr = []
        scaled_lr = lr

        for param_key, override in mup_overrides.items():
            scaled_lr = override.get('max_lr', lr)
            predicate_fn = param_key.with_name_predicate.fn

            for pname, param in model.named_parameters():
                if predicate_fn(param, pname):
                    params_with_scaled_lr.append(param)
                else:
                    params_with_base_lr.append(param)
            break

        param_groups = []
        if params_with_scaled_lr:
            param_groups.append({'params': params_with_scaled_lr, 'lr': scaled_lr})
        if params_with_base_lr:
            param_groups.append({'params': params_with_base_lr, 'lr': lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    vocab_size = data_provider.vocab_size

    # Results: std(x_t - x_0) for each metric at each step
    results = {
        'logits': [],
        'attn_logits': [],
        'word_embedding': [],
    }

    # Use fixed batch for consistency
    fixed_batch = data_provider.get_batch()

    for step in range(num_steps + 1):  # +1 to include t=0
        current_activations = {}

        # Forward pass to collect activations
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(
                input_ids=fixed_batch['input_ids'],
                position_ids=fixed_batch['position_ids'],
                attention_mask=None,
            )

        current_activations['logits'] = logits.detach().clone().float()

        if step == 0:
            # Store initial activations
            initial_activations = {k: v.clone() for k, v in current_activations.items()}
            # Record zeros for t=0
            for key in results:
                results[key].append(0.0)
        else:
            # Compute std(x_t - x_0) for each metric
            for key in results:
                if key in current_activations and key in initial_activations:
                    delta = current_activations[key] - initial_activations[key]
                    results[key].append(delta.std().item())
                else:
                    results[key].append(0.0)

        # Training step (skip for t=0)
        if step < num_steps:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                train_logits = model(
                    input_ids=fixed_batch['input_ids'],
                    position_ids=fixed_batch['position_ids'],
                    attention_mask=None,
                )
                loss_flat = F.cross_entropy(
                    train_logits.view(-1, vocab_size).float(),
                    fixed_batch['labels'].view(-1),
                    reduction='none',
                )
                loss_mask_flat = fixed_batch['loss_mask'].view(-1)
                loss = (loss_flat * loss_mask_flat).sum() / loss_mask_flat.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

    return results


def run_delta_coordinate_check(
    widths: List[int],
    base_hidden_size: int,
    num_layers: int = 4,
    num_steps: int = 4,
    num_seeds: int = 3,
    vocab_size: int = 256,
    data_provider=None,
    lr: float = 1e-3,
) -> Tuple[Dict[int, Dict[str, Dict[str, List[float]]]], Dict[int, Dict[str, Dict[str, List[float]]]]]:
    """Run coordinate check tracking std(x_t - x_0) across widths with multiple seeds.

    Returns:
        Tuple of (results_std, results_mup) where each is:
        Dict[width, Dict[metric, {'mean': List[float], 'std': List[float]}]]
    """
    print("\n" + "=" * 60)
    print("Running Delta-from-Init Coordinate Check")
    print("=" * 60)
    print(f"Widths: {widths}")
    print(f"Base hidden size: {base_hidden_size}")
    print(f"Num steps (t values): {num_steps}")
    print(f"Num seeds: {num_seeds}")

    results_std = {}
    results_mup = {}

    for width in widths:
        num_heads = max(1, width // 64)
        print(f"\nWidth {width}:")

        # Collect results across seeds
        std_seed_results = []  # List of dicts per seed
        mup_seed_results = []

        for seed in range(num_seeds):
            # Standard parameterization
            print(f"  Seed {seed+1}/{num_seeds} Standard...", end=" ", flush=True)
            torch.manual_seed(seed * 1000 + width)
            data_provider.reset()
            model_std, config_std = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=False,
                base_hidden_size=None,
                vocab_size=vocab_size,
            )
            std_seed_results.append(collect_delta_from_init_stats(
                model_std, config_std, data_provider, num_steps=num_steps, lr=lr
            ))
            del model_std
            torch.cuda.empty_cache()
            print("done")

            # MuP
            print(f"  Seed {seed+1}/{num_seeds} MuP...", end=" ", flush=True)
            torch.manual_seed(seed * 1000 + width)
            data_provider.reset()
            model_mup, config_mup = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=True,
                base_hidden_size=base_hidden_size,
                vocab_size=vocab_size,
            )
            mup_seed_results.append(collect_delta_from_init_stats(
                model_mup, config_mup, data_provider, num_steps=num_steps, lr=lr
            ))
            del model_mup
            torch.cuda.empty_cache()
            print("done")

        # Aggregate across seeds: compute mean and std for each metric at each step
        metrics = ['logits', 'attn_logits', 'word_embedding']

        results_std[width] = {}
        results_mup[width] = {}

        for metric in metrics:
            # Stack values across seeds: shape (num_seeds, num_steps+1)
            std_vals = np.array([r[metric] for r in std_seed_results])
            mup_vals = np.array([r[metric] for r in mup_seed_results])

            results_std[width][metric] = {
                'mean': std_vals.mean(axis=0).tolist(),
                'std': std_vals.std(axis=0).tolist(),
            }
            results_mup[width][metric] = {
                'mean': mup_vals.mean(axis=0).tolist(),
                'std': mup_vals.std(axis=0).tolist(),
            }

    return results_std, results_mup


def plot_delta_coordinate_check(
    results_std: Dict[int, Dict[str, Dict[str, List[float]]]],
    results_mup: Dict[int, Dict[str, Dict[str, List[float]]]],
    output_dir: str,
    base_width: int = 128,
):
    """Plot 1x3 grid showing std(x_t - x_0) vs width with SP and MuP overlaid.

    SP uses red/orange color scheme, MuP uses blue/purple color scheme.
    This allows direct visual comparison on the same axes.
    """
    widths = sorted(results_std.keys())
    metrics = ['logits', 'attn_logits', 'word_embedding']
    metric_labels = ['logits', 'attn logits', 'word embedding']

    # Get number of time steps from data
    num_steps = len(results_std[widths[0]]['logits']['mean'])

    # SP color palette: light orange → dark red
    sp_colors = [
        '#ffd4a3',  # light peach
        '#ffb366',  # orange
        '#e67300',  # dark orange
        '#cc4400',  # red-orange
        '#992200',  # dark red
    ]
    # MuP color palette: light blue → dark purple
    mup_colors = [
        '#a3d4ff',  # light blue
        '#66a3ff',  # blue
        '#3366cc',  # medium blue
        '#6633cc',  # purple
        '#330099',  # dark purple
    ]

    # Extend or truncate to match num_steps
    if num_steps <= len(sp_colors):
        sp_colors = sp_colors[:num_steps]
        mup_colors = mup_colors[:num_steps]
    else:
        # Interpolate if more steps needed
        sp_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('sp', sp_colors)
        mup_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('mup', mup_colors)
        sp_colors = [sp_cmap(i / (num_steps - 1)) for i in range(num_steps)]
        mup_colors = [mup_cmap(i / (num_steps - 1)) for i in range(num_steps)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]

        # Plot SP (solid lines, red/orange)
        for t in range(num_steps):
            y_mean = [results_std[w][metric]['mean'][t] for w in widths]
            y_std = [results_std[w][metric]['std'][t] for w in widths]
            color = sp_colors[t]

            ax.plot(widths, y_mean, '-', color=color, linewidth=2.0,
                   label=f'SP t={t}' if col == 0 else None)
            y_upper = [m + s for m, s in zip(y_mean, y_std)]
            y_lower = [m - s for m, s in zip(y_mean, y_std)]
            ax.fill_between(widths, y_lower, y_upper, color=color, alpha=0.2)

        # Plot MuP (dashed lines, blue/purple)
        for t in range(num_steps):
            y_mean = [results_mup[w][metric]['mean'][t] for w in widths]
            y_std = [results_mup[w][metric]['std'][t] for w in widths]
            color = mup_colors[t]

            ax.plot(widths, y_mean, '--', color=color, linewidth=2.0,
                   label=f'MuP t={t}' if col == 0 else None)
            y_upper = [m + s for m, s in zip(y_mean, y_std)]
            y_lower = [m - s for m, s in zip(y_mean, y_std)]
            ax.fill_between(widths, y_lower, y_upper, color=color, alpha=0.2)

        # Formatting
        ax.set_facecolor('#f5f5f5')
        ax.grid(True, alpha=0.5, color='white', linewidth=0.8)
        ax.tick_params(labelsize=10)
        ax.set_xlabel('width', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        if col == 0:
            ax.set_ylabel('std(x$_t$ − x$_0$)', fontsize=11)

    # Add legend to first subplot
    axes[0].legend(loc='upper left', fontsize=8, ncol=2,
                   framealpha=0.9, edgecolor='lightgray')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mup_coord_check_delta.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mup_coord_check_delta.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nDelta coordinate check plot saved to {output_dir}/mup_coord_check_delta.png")

    # Print summary
    print("\nSummary (std(x_t - x_0) at final step):")
    print(f"{'Width':<10} {'SP logits':<12} {'MuP logits':<12} {'Ratio':<10}")
    print("-" * 44)
    for w in widths:
        sp_val = results_std[w]['logits']['mean'][-1]
        mup_val = results_mup[w]['logits']['mean'][-1]
        ratio = sp_val / mup_val if mup_val > 0 else float('inf')
        print(f"{w:<10} {sp_val:<12.4f} {mup_val:<12.4f} {ratio:<10.2f}")


def run_coordinate_check(
    widths: List[int],
    base_hidden_size: int,
    num_layers: int = 4,
    num_seeds: int = 3,
    num_steps: int = 10,
    vocab_size: int = 50257,
    data_provider=None,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Run coordinate check across multiple widths and seeds.

    Now supports per-step tracking during training to match Microsoft's
    mup.coord_check.get_coord_data() approach.

    Args:
        widths: List of hidden sizes to test
        base_hidden_size: Base width for MuP scaling
        num_layers: Number of transformer layers
        num_seeds: Number of random seeds for init check
        num_steps: Number of training steps for per-step tracking
        vocab_size: Vocabulary size
        data_provider: DataProvider for training data (required for per-step tracking)

    Returns:
        Tuple of (results_std, results_mup, step_stats_std, step_stats_mup)
        - results_std/mup: Dict[width, List[logits_std]] for init check
        - step_stats_std/mup: Dict[width, List[step_stats]] for per-step tracking
    """
    print("\n" + "=" * 60)
    print("Running Coordinate Check")
    print("=" * 60)
    print(f"Widths: {widths}")
    print(f"Base hidden size: {base_hidden_size}")
    print(f"Num layers: {num_layers}")
    print(f"Num seeds: {num_seeds}")
    print(f"Num steps (per-step tracking): {num_steps}")
    print(f"Vocab size: {vocab_size}")

    results_std = {w: [] for w in widths}
    results_mup = {w: [] for w in widths}
    step_stats_std = {w: [] for w in widths}
    step_stats_mup = {w: [] for w in widths}

    # Part 1: Init-time coordinate check (original behavior)
    print("\n--- Init-Time Coordinate Check ---")
    for seed in range(num_seeds):
        print(f"\nSeed {seed + 1}/{num_seeds}")
        for width in widths:
            num_heads = max(1, width // 64)
            print(f"  Width {width}, heads {num_heads}...", end=" ", flush=True)

            # Standard model
            torch.manual_seed(seed * 1000 + width)
            model_std, _ = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=False,
                base_hidden_size=None,
                vocab_size=vocab_size,
            )
            stats_std = collect_activation_stats(model_std, None)
            results_std[width].append(stats_std['logits']['std'])
            del model_std
            torch.cuda.empty_cache()

            # MuP model
            torch.manual_seed(seed * 1000 + width)
            model_mup, _ = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=True,
                base_hidden_size=base_hidden_size,
                vocab_size=vocab_size,
            )
            stats_mup = collect_activation_stats(model_mup, None)
            results_mup[width].append(stats_mup['logits']['std'])
            del model_mup
            torch.cuda.empty_cache()

            print(f"std={results_std[width][-1]:.4f}, mup={results_mup[width][-1]:.4f}")

    # Part 2: Per-step tracking during training
    if data_provider is not None and num_steps > 0:
        print("\n--- Per-Step Activation Tracking ---")
        for width in widths:
            num_heads = max(1, width // 64)
            print(f"\nWidth {width} (training for {num_steps} steps):")

            # Standard model
            print("  Standard...", end=" ", flush=True)
            torch.manual_seed(42)
            data_provider.reset()
            model_std, config_std = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=False,
                base_hidden_size=None,
                vocab_size=vocab_size,
            )
            step_stats_std[width] = collect_activation_stats_per_step(
                model_std, config_std, data_provider, num_steps=num_steps
            )
            del model_std
            torch.cuda.empty_cache()
            print("done")

            # MuP model
            print("  MuP...", end=" ", flush=True)
            torch.manual_seed(42)
            data_provider.reset()
            model_mup, config_mup = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=True,
                base_hidden_size=base_hidden_size,
                vocab_size=vocab_size,
            )
            step_stats_mup[width] = collect_activation_stats_per_step(
                model_mup, config_mup, data_provider, num_steps=num_steps
            )
            del model_mup
            torch.cuda.empty_cache()
            print("done")

    return results_std, results_mup, step_stats_std, step_stats_mup


def run_lr_sweep(
    widths: List[int],
    lrs: List[float],
    base_hidden_size: int,
    num_layers: int = 4,
    num_steps: int = 100,
    use_mup: bool = True,
    data_provider: Optional[DataProvider] = None,
    debug_gradients: bool = False,
) -> Dict[int, Dict[float, float]]:
    """Run LR sweep for each width, return final losses."""
    from megatron.core.optimizer import get_mup_config_overrides
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    print("\n" + "=" * 60)
    print(f"Running LR Sweep ({'MuP' if use_mup else 'Standard'})")
    print("=" * 60)
    print(f"Widths: {widths}")
    print(f"LRs: {lrs}")
    print(f"Steps per run: {num_steps}")

    results = {w: {} for w in widths}

    # Get vocab size from data provider
    vocab_size = data_provider.vocab_size if data_provider else 50257

    for width in widths:
        num_heads = max(1, width // 64)
        print(f"\nWidth {width}:")

        for lr in lrs:
            torch.manual_seed(42)
            if data_provider:
                data_provider.reset()

            model, config = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=use_mup,
                base_hidden_size=base_hidden_size,
                vocab_size=vocab_size,
            )

            # Setup optimizer with MuP LR scaling
            opt_config = OptimizerConfig(lr=lr, min_lr=lr / 10)

            if use_mup and config.mup_width_mult != 1.0:
                mup_overrides = get_mup_config_overrides(
                    opt_config,
                    mup_width_mult=config.mup_width_mult,
                    optimizer_type='adam'
                )

                params_with_scaled_lr = []
                params_with_base_lr = []
                scaled_lr = lr

                for param_key, override in mup_overrides.items():
                    scaled_lr = override.get('max_lr', lr)
                    predicate_fn = param_key.with_name_predicate.fn

                    for name, param in model.named_parameters():
                        if predicate_fn(param, name):
                            params_with_scaled_lr.append(param)
                        else:
                            params_with_base_lr.append(param)
                    break

                param_groups = []
                if params_with_scaled_lr:
                    param_groups.append({'params': params_with_scaled_lr, 'lr': scaled_lr})
                if params_with_base_lr:
                    param_groups.append({'params': params_with_base_lr, 'lr': lr})

                # DEBUG: Print parameter group info for first LR only
                if lr == lrs[0]:
                    hidden_count = sum(p.numel() for p in params_with_scaled_lr)
                    embed_count = sum(p.numel() for p in params_with_base_lr)
                    total_count = hidden_count + embed_count
                    print(f"  MuP param groups (width_mult={config.mup_width_mult:.1f}):")
                    print(f"    Hidden layers: {len(params_with_scaled_lr)} tensors, {hidden_count:,} params ({100*hidden_count/total_count:.1f}%) @ lr={scaled_lr:.2e}")
                    print(f"    Embed/output:  {len(params_with_base_lr)} tensors, {embed_count:,} params ({100*embed_count/total_count:.1f}%) @ lr={lr:.2e}")

                    # Show which params have the is_embedding_parameter attribute
                    print(f"    Params with is_embedding_parameter attribute:")
                    for name, param in model.named_parameters():
                        attr_val = getattr(param, 'is_embedding_parameter', None)
                        if attr_val is not None:
                            print(f"      {name}: is_embedding_parameter={attr_val}")
            else:
                param_groups = [{'params': model.parameters(), 'lr': lr}]

            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

            # Training loop
            model.train()
            losses = []

            for step in range(num_steps):
                if data_provider:
                    batch = data_provider.get_batch()
                    input_ids = batch['input_ids']
                    position_ids = batch['position_ids']
                    labels = batch['labels']
                    loss_mask = batch['loss_mask']
                else:
                    # Fallback to random data
                    batch_size, seq_len = 4, 128
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
                    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
                    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
                    loss_mask = torch.ones_like(labels, dtype=torch.float)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=None,
                    )
                    # Apply loss mask for proper loss computation
                    loss_flat = F.cross_entropy(
                        logits.view(-1, vocab_size).float(),
                        labels.view(-1),
                        reduction='none',
                    )
                    loss_mask_flat = loss_mask.view(-1)
                    loss = (loss_flat * loss_mask_flat).sum() / loss_mask_flat.sum()

                loss.backward()

                # DEBUG: Track per-group gradient norms
                if debug_gradients and use_mup and config.mup_width_mult != 1.0 and step == 0:
                    hidden_grad_norm = torch.sqrt(sum(
                        p.grad.float().pow(2).sum() for p in params_with_scaled_lr if p.grad is not None
                    )).item()
                    embed_grad_norm = torch.sqrt(sum(
                        p.grad.float().pow(2).sum() for p in params_with_base_lr if p.grad is not None
                    )).item()
                    print(f"    Step 0 grad norms: hidden={hidden_grad_norm:.4f}, embed/out={embed_grad_norm:.4f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())

            final_loss = np.mean(losses[-10:])
            results[width][lr] = final_loss
            print(f"  LR {lr:.0e}: final_loss={final_loss:.4f}")

            del model, optimizer
            torch.cuda.empty_cache()

    return results


def run_loss_curves(
    widths: List[int],
    base_hidden_size: int,
    base_lr: float,
    num_layers: int = 4,
    num_steps: int = 500,
    data_provider: Optional[DataProvider] = None,
    debug_gradients: bool = False,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """Run training and collect loss curves for standard and MuP."""
    from megatron.core.optimizer import get_mup_config_overrides
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    print("\n" + "=" * 60)
    print("Running Loss Curve Comparison")
    print("=" * 60)
    print(f"Widths: {widths}")
    print(f"Base LR: {base_lr}")
    print(f"Steps: {num_steps}")

    results_std = {}
    results_mup = {}

    # Get vocab size from data provider
    vocab_size = data_provider.vocab_size if data_provider else 50257

    for width in widths:
        num_heads = max(1, width // 64)
        print(f"\nWidth {width}:")

        for use_mup, results, label in [(False, results_std, "Standard"), (True, results_mup, "MuP")]:
            torch.manual_seed(42)
            np.random.seed(42)
            if data_provider:
                data_provider.reset()

            model, config = create_gpt_model(
                hidden_size=width,
                num_layers=num_layers,
                num_heads=num_heads,
                use_mup=use_mup,
                base_hidden_size=base_hidden_size,
                vocab_size=vocab_size,
            )

            # Setup optimizer
            opt_config = OptimizerConfig(lr=base_lr, min_lr=base_lr / 10)

            if use_mup and config.mup_width_mult != 1.0:
                mup_overrides = get_mup_config_overrides(
                    opt_config,
                    mup_width_mult=config.mup_width_mult,
                    optimizer_type='adam'
                )

                params_with_scaled_lr = []
                params_with_base_lr = []
                scaled_lr = base_lr

                for param_key, override in mup_overrides.items():
                    scaled_lr = override.get('max_lr', base_lr)
                    predicate_fn = param_key.with_name_predicate.fn

                    for name, param in model.named_parameters():
                        if predicate_fn(param, name):
                            params_with_scaled_lr.append(param)
                        else:
                            params_with_base_lr.append(param)
                    break

                param_groups = []
                if params_with_scaled_lr:
                    param_groups.append({'params': params_with_scaled_lr, 'lr': scaled_lr})
                if params_with_base_lr:
                    param_groups.append({'params': params_with_base_lr, 'lr': base_lr})

                # DEBUG: Print parameter group info
                hidden_count = sum(p.numel() for p in params_with_scaled_lr)
                embed_count = sum(p.numel() for p in params_with_base_lr)
                total_count = hidden_count + embed_count
                print(f"  {label} MuP param groups (width_mult={config.mup_width_mult:.1f}):")
                print(f"    Hidden layers: {len(params_with_scaled_lr)} tensors, {hidden_count:,} params ({100*hidden_count/total_count:.1f}%) @ lr={scaled_lr:.2e}")
                print(f"    Embed/output:  {len(params_with_base_lr)} tensors, {embed_count:,} params ({100*embed_count/total_count:.1f}%) @ lr={base_lr:.2e}")
            else:
                param_groups = [{'params': model.parameters(), 'lr': base_lr}]

            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

            # Training loop
            model.train()
            losses = []

            for step in range(num_steps):
                if data_provider:
                    batch = data_provider.get_batch()
                    input_ids = batch['input_ids']
                    position_ids = batch['position_ids']
                    labels = batch['labels']
                    loss_mask = batch['loss_mask']
                else:
                    # Fallback to random data with fixed seed for reproducibility
                    torch.manual_seed(12345 + step)
                    batch_size, seq_len = 4, 128
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
                    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
                    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
                    loss_mask = torch.ones_like(labels, dtype=torch.float)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=None,
                    )
                    # Apply loss mask for proper loss computation
                    loss_flat = F.cross_entropy(
                        logits.view(-1, vocab_size).float(),
                        labels.view(-1),
                        reduction='none',
                    )
                    loss_mask_flat = loss_mask.view(-1)
                    loss = (loss_flat * loss_mask_flat).sum() / loss_mask_flat.sum()

                loss.backward()

                # DEBUG: Track per-group gradient norms
                if debug_gradients and use_mup and config.mup_width_mult != 1.0 and step == 0:
                    hidden_grad_norm = torch.sqrt(sum(
                        p.grad.float().pow(2).sum() for p in params_with_scaled_lr if p.grad is not None
                    )).item()
                    embed_grad_norm = torch.sqrt(sum(
                        p.grad.float().pow(2).sum() for p in params_with_base_lr if p.grad is not None
                    )).item()
                    print(f"    Step 0 grad norms: hidden={hidden_grad_norm:.4f}, embed/out={embed_grad_norm:.4f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())

                if step % 100 == 0:
                    print(f"  {label} step {step}: loss={loss.item():.4f}")

            results[width] = losses

            del model, optimizer
            torch.cuda.empty_cache()

    return results_std, results_mup


def plot_coordinate_check(
    results_std: Dict[int, List[float]],
    results_mup: Dict[int, List[float]],
    output_dir: str,
):
    """Plot coordinate check results."""
    widths = sorted(results_std.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute means and stds
    std_means = [np.mean(results_std[w]) for w in widths]
    std_stds = [np.std(results_std[w]) for w in widths]
    mup_means = [np.mean(results_mup[w]) for w in widths]
    mup_stds = [np.std(results_mup[w]) for w in widths]

    # Left plot: Raw values
    ax1 = axes[0]
    ax1.errorbar(widths, std_means, yerr=std_stds, marker='o', markersize=8,
                 capsize=5, label='Standard', color='#e74c3c', linewidth=2)
    ax1.errorbar(widths, mup_means, yerr=mup_stds, marker='s', markersize=8,
                 capsize=5, label='MuP', color='#2ecc71', linewidth=2)
    ax1.set_xlabel('Width (hidden_size)', fontsize=12)
    ax1.set_ylabel('Logits Std', fontsize=12)
    ax1.set_title('Coordinate Check: Output Stability', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xticks(widths)

    # Right plot: Normalized to base width
    ax2 = axes[1]
    base_std = std_means[0]
    base_mup = mup_means[0]
    std_normalized = [s / base_std for s in std_means]
    mup_normalized = [m / base_mup for m in mup_means]

    ax2.plot(widths, std_normalized, marker='o', markersize=8,
             label='Standard', color='#e74c3c', linewidth=2)
    ax2.plot(widths, mup_normalized, marker='s', markersize=8,
             label='MuP', color='#2ecc71', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal (flat)')
    ax2.set_xlabel('Width (hidden_size)', fontsize=12)
    ax2.set_ylabel('Normalized Logits Std (relative to base width)', fontsize=12)
    ax2.set_title('Coordinate Check: Normalized', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks(widths)

    # Add CV annotations
    cv_std = np.std(std_means) / np.mean(std_means)
    cv_mup = np.std(mup_means) / np.mean(mup_means)
    ax1.text(0.05, 0.95, f'CV(Std)={cv_std:.3f}\nCV(MuP)={cv_mup:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coordinate_check.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'coordinate_check.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nCoordinate check plot saved to {output_dir}/coordinate_check.png")
    print(f"CV(Standard) = {cv_std:.4f}")
    print(f"CV(MuP) = {cv_mup:.4f}")
    print(f"Improvement: {cv_std/cv_mup:.1f}x")


def plot_coord_check_per_step(
    step_stats_std: Dict[int, List[Dict]],
    step_stats_mup: Dict[int, List[Dict]],
    output_dir: str,
):
    """Plot per-step activation statistics for coordinate check.

    Creates Microsoft-style coord check plots showing activation norms
    across training steps for each width. MuP should show flat/parallel
    lines while standard parameterization should show divergence.

    Args:
        step_stats_std: Dict[width, List[step_stats]] for standard param
        step_stats_mup: Dict[width, List[step_stats]] for MuP
        output_dir: Directory to save plots
    """
    widths = sorted(step_stats_std.keys())

    # Skip if no per-step data
    if not step_stats_std or not step_stats_std[widths[0]]:
        print("No per-step data available, skipping per-step plot")
        return

    # Get all tracked layer names from first width's first step
    sample_stats = step_stats_std[widths[0]][0]
    layer_names = list(sample_stats.get('layers', {}).keys())

    # Add logits as a tracked "layer"
    tracked_metrics = layer_names + ['logits']

    # Create a figure with subplots for each tracked layer
    # Arrange in 2 columns
    n_metrics = len(tracked_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(widths)))

    for idx, metric in enumerate(tracked_metrics):
        row = idx // n_cols
        col_base = (idx % n_cols) * 2

        # Left: Standard parameterization
        ax_std = axes[row, col_base]
        # Right: MuP
        ax_mup = axes[row, col_base + 1]

        for i, width in enumerate(widths):
            steps = [s['step'] for s in step_stats_std[width]]

            # Get L1 values for this metric
            if metric == 'logits':
                std_l1 = [s['logits_l1'] for s in step_stats_std[width]]
                mup_l1 = [s['logits_l1'] for s in step_stats_mup[width]]
            else:
                std_l1 = [s['layers'].get(metric, {}).get('l1', 0) for s in step_stats_std[width]]
                mup_l1 = [s['layers'].get(metric, {}).get('l1', 0) for s in step_stats_mup[width]]

            ax_std.plot(steps, std_l1, label=f'w={width}', color=colors[i], linewidth=2)
            ax_mup.plot(steps, mup_l1, label=f'w={width}', color=colors[i], linewidth=2)

        ax_std.set_title(f'{metric} - Standard', fontsize=10, fontweight='bold')
        ax_mup.set_title(f'{metric} - MuP', fontsize=10, fontweight='bold')

        ax_std.set_xlabel('Step', fontsize=9)
        ax_std.set_ylabel('L1 Norm', fontsize=9)
        ax_mup.set_xlabel('Step', fontsize=9)
        ax_mup.set_ylabel('L1 Norm', fontsize=9)

        ax_std.grid(True, alpha=0.3)
        ax_mup.grid(True, alpha=0.3)

        if idx == 0:
            ax_std.legend(fontsize=8, loc='upper right')
            ax_mup.legend(fontsize=8, loc='upper right')

    # Hide any unused subplots
    total_cells = n_rows * n_cols * 2
    used_cells = n_metrics * 2
    for idx in range(used_cells, total_cells):
        row = idx // (n_cols * 2)
        col = idx % (n_cols * 2)
        if row < n_rows:
            axes[row, col].axis('off')

    plt.suptitle('Per-Step Coordinate Check: Activation L1 Norms During Training\n'
                 '(MuP should show parallel lines; Standard should diverge)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coord_check_per_step.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'coord_check_per_step.pdf'), bbox_inches='tight')
    plt.close()

    # Also create a summary plot showing CV across widths at each step
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Compute CV at each step for logits L1
    num_steps = len(step_stats_std[widths[0]])
    steps = list(range(num_steps))

    cv_std_per_step = []
    cv_mup_per_step = []

    for step in steps:
        std_vals = [step_stats_std[w][step]['logits_l1'] for w in widths]
        mup_vals = [step_stats_mup[w][step]['logits_l1'] for w in widths]

        cv_std_per_step.append(np.std(std_vals) / np.mean(std_vals) if np.mean(std_vals) > 0 else 0)
        cv_mup_per_step.append(np.std(mup_vals) / np.mean(mup_vals) if np.mean(mup_vals) > 0 else 0)

    # Plot 1: CV over steps
    ax1 = axes2[0]
    ax1.plot(steps, cv_std_per_step, 'o-', label='Standard', color='#e74c3c', linewidth=2, markersize=4)
    ax1.plot(steps, cv_mup_per_step, 's-', label='MuP', color='#2ecc71', linewidth=2, markersize=4)
    ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='CV=0.1 threshold')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('CV (Coefficient of Variation) across widths', fontsize=12)
    ax1.set_title('Width Invariance During Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: Logits L1 norm for all widths
    ax2 = axes2[1]
    for i, width in enumerate(widths):
        std_l1 = [s['logits_l1'] for s in step_stats_std[width]]
        mup_l1 = [s['logits_l1'] for s in step_stats_mup[width]]
        ax2.plot(steps, std_l1, '--', label=f'Std w={width}', color=colors[i], linewidth=1.5, alpha=0.7)
        ax2.plot(steps, mup_l1, '-', label=f'MuP w={width}', color=colors[i], linewidth=2)

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Logits L1 Norm', fontsize=12)
    ax2.set_title('Logits Activation Scale\n(Solid=MuP, Dashed=Standard)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coord_check_per_step_summary.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'coord_check_per_step_summary.pdf'), bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print(f"\nPer-step coordinate check plots saved to {output_dir}/")
    print(f"  - coord_check_per_step.png (detailed per-layer)")
    print(f"  - coord_check_per_step_summary.png (CV summary)")
    print(f"\nCV Summary (Logits L1):")
    print(f"  Step 0:  Standard CV={cv_std_per_step[0]:.4f}, MuP CV={cv_mup_per_step[0]:.4f}")
    print(f"  Step {num_steps-1}: Standard CV={cv_std_per_step[-1]:.4f}, MuP CV={cv_mup_per_step[-1]:.4f}")
    mean_cv_std = np.mean(cv_std_per_step)
    mean_cv_mup = np.mean(cv_mup_per_step)
    print(f"  Mean:    Standard CV={mean_cv_std:.4f}, MuP CV={mean_cv_mup:.4f}")
    if mean_cv_mup > 0:
        print(f"  Improvement: {mean_cv_std/mean_cv_mup:.1f}x")


def plot_lr_sweep(
    results_std: Dict[int, Dict[float, float]],
    results_mup: Dict[int, Dict[float, float]],
    output_dir: str,
):
    """Plot LR sweep results matching MuP paper Figure 1 style.

    Shows Training Loss vs log₂(LearningRate) with:
    - Light pink → dark purple color gradient by width
    - "optimum shifts" annotation for SP, "optimum stable" for MuP
    - Dashed horizontal reference line at best MuP loss
    """
    widths = sorted(results_std.keys())
    lrs = sorted(results_std[widths[0]].keys())

    # Convert LRs to log2 for x-axis
    log2_lrs = [np.log2(lr) for lr in lrs]

    # Color palette: light pink → dark purple (matching paper)
    color_list = [
        '#f5d0c5',  # lightest (smallest width)
        '#e8b4b4',
        '#d4a5a5',
        '#c08090',
        '#a67c94',
        '#8a6080',
        '#6b4c7a',
        '#4a3060',
        '#2d1e3e',  # darkest (largest width)
    ]
    # Interpolate colors for number of widths
    if len(widths) <= len(color_list):
        colors = color_list[:len(widths)]
    else:
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('mup', color_list)
        colors = [cmap(i / (len(widths) - 1)) for i in range(len(widths))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Find best MuP loss for reference line
    best_mup_loss = float('inf')
    for width in widths:
        for lr in lrs:
            if results_mup[width][lr] < best_mup_loss:
                best_mup_loss = results_mup[width][lr]

    for ax_idx, (results, title, is_mup) in enumerate([
        (results_std, 'Standard Practice', False),
        (results_mup, 'Our Work', True)
    ]):
        ax = axes[ax_idx]

        # Plot each width
        optimal_lrs = []
        optimal_losses = []
        for i, width in enumerate(widths):
            losses = [results[width][lr] for lr in lrs]
            ax.plot(log2_lrs, losses, '-', color=colors[i], linewidth=1.8,
                   label=f'{width}' if ax_idx == 0 else None)

            # Track optimal LR
            min_idx = np.argmin(losses)
            optimal_lrs.append(log2_lrs[min_idx])
            optimal_losses.append(losses[min_idx])

            # Mark the minimum with a star marker
            ax.plot(log2_lrs[min_idx], losses[min_idx], '*', color=colors[i],
                   markersize=12, markeredgecolor='black', markeredgewidth=0.5)

        # Add dashed reference line at best MuP loss
        ax.axhline(y=best_mup_loss, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)

        # Formatting to match paper
        ax.set_facecolor('#e8e8f0')
        ax.grid(True, alpha=0.5, color='white', linewidth=0.8)
        ax.set_xlabel('log₂LearningRate', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel('Training Loss', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=9)

        # Add "optimum shifts/stable" annotation
        if is_mup:
            # Arrow pointing right with "optimum stable"
            ax.annotate('optimum stable →',
                       xy=(max(optimal_lrs), best_mup_loss + 0.1),
                       fontsize=9, ha='center')
        else:
            # Arrow pointing at the shifting optimum
            mid_idx = len(widths) // 2
            ax.annotate('↑\noptimum shifts',
                       xy=(optimal_lrs[mid_idx], optimal_losses[mid_idx] + 0.3),
                       fontsize=9, ha='center', va='bottom')

    # Add legend to first subplot
    axes[0].legend(loc='upper left', fontsize=8, title='Width',
                   framealpha=0.9, edgecolor='lightgray')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_sweep.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'lr_sweep.pdf'), bbox_inches='tight')
    plt.close()

    # Print summary
    print(f"\nLR sweep plot saved to {output_dir}/lr_sweep.png")
    print("\nOptimal LRs by width:")
    print(f"{'Width':<10} {'SP opt LR':<15} {'MuP opt LR':<15}")
    print("-" * 40)
    for width in widths:
        sp_opt = min(lrs, key=lambda lr: results_std[width][lr])
        mup_opt = min(lrs, key=lambda lr: results_mup[width][lr])
        print(f"{width:<10} {sp_opt:<15.2e} {mup_opt:<15.2e}")


def plot_loss_curves(
    results_std: Dict[int, List[float]],
    results_mup: Dict[int, List[float]],
    output_dir: str,
):
    """Plot loss curves comparison."""
    widths = sorted(results_std.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(widths)))

    # Standard
    ax1 = axes[0]
    for i, width in enumerate(widths):
        losses = results_std[width]
        # Smooth with rolling average
        window = 20
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(len(smoothed)), smoothed, label=f'width={width}', color=colors[i], linewidth=2)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Standard Parameterization\n(Curves diverge across widths)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # MuP
    ax2 = axes[1]
    for i, width in enumerate(widths):
        losses = results_mup[width]
        window = 20
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(len(smoothed)), smoothed, label=f'width={width}', color=colors[i], linewidth=2)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('MuP\n(Curves align across widths)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'loss_curves.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nLoss curves plot saved to {output_dir}/loss_curves.png")


def plot_summary(
    coord_std: Dict[int, List[float]],
    coord_mup: Dict[int, List[float]],
    lr_std: Dict[int, Dict[float, float]],
    lr_mup: Dict[int, Dict[float, float]],
    output_dir: str,
):
    """Create a summary plot combining key results."""
    widths = sorted(coord_std.keys())
    lrs = sorted(lr_std[widths[0]].keys())

    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Coordinate check
    ax1 = fig.add_subplot(2, 2, 1)
    std_means = [np.mean(coord_std[w]) for w in widths]
    mup_means = [np.mean(coord_mup[w]) for w in widths]

    x = np.arange(len(widths))
    width_bar = 0.35
    ax1.bar(x - width_bar/2, std_means, width_bar, label='Standard', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width_bar/2, mup_means, width_bar, label='MuP', color='#2ecc71', alpha=0.8)
    ax1.set_xlabel('Width', fontsize=12)
    ax1.set_ylabel('Logits Std', fontsize=12)
    ax1.set_title('Output Stability', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(widths)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Normalized coordinate check
    ax2 = fig.add_subplot(2, 2, 2)
    base_std = std_means[0]
    base_mup = mup_means[0]
    std_norm = [s / base_std for s in std_means]
    mup_norm = [m / base_mup for m in mup_means]

    ax2.plot(widths, std_norm, 'o-', label='Standard', color='#e74c3c', linewidth=2, markersize=8)
    ax2.plot(widths, mup_norm, 's-', label='MuP', color='#2ecc71', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Width', fontsize=12)
    ax2.set_ylabel('Normalized Output Std', fontsize=12)
    ax2.set_title('Output Stability (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks(widths)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Optimal LR vs width
    ax3 = fig.add_subplot(2, 2, 3)

    optimal_lrs_std = []
    optimal_lrs_mup = []
    for width in widths:
        std_losses = {lr: lr_std[width][lr] for lr in lrs}
        mup_losses = {lr: lr_mup[width][lr] for lr in lrs}
        optimal_lrs_std.append(min(std_losses, key=std_losses.get))
        optimal_lrs_mup.append(min(mup_losses, key=mup_losses.get))

    ax3.plot(widths, optimal_lrs_std, 'o-', label='Standard', color='#e74c3c', linewidth=2, markersize=8)
    ax3.plot(widths, optimal_lrs_mup, 's-', label='MuP', color='#2ecc71', linewidth=2, markersize=8)
    ax3.set_xlabel('Width', fontsize=12)
    ax3.set_ylabel('Optimal Learning Rate', fontsize=12)
    ax3.set_title('Optimal LR vs Width', fontsize=14, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.set_xticks(widths)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Summary metrics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    cv_std = np.std(std_means) / np.mean(std_means)
    cv_mup = np.std(mup_means) / np.mean(mup_means)
    lr_spread_std = max(optimal_lrs_std) / min(optimal_lrs_std)
    lr_spread_mup = max(optimal_lrs_mup) / min(optimal_lrs_mup)

    summary_text = f"""
    MuP Verification Summary
    ========================

    Coordinate Check (Output Stability):
      Standard CV: {cv_std:.4f}
      MuP CV:      {cv_mup:.4f}
      Improvement: {cv_std/cv_mup:.1f}x

    LR Transfer (Optimal LR Spread):
      Standard: {lr_spread_std:.2f}x spread
      MuP:      {lr_spread_mup:.2f}x spread

    Result: {'PASS' if cv_mup < 0.1 and lr_spread_mup < 2.0 else 'NEEDS REVIEW'}

    A correct MuP implementation should have:
    - CV < 0.1 (output stable across widths)
    - LR spread < 2x (same optimal LR works)
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='gray'))

    plt.suptitle('MuP Implementation Verification - Megatron-LM', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mup_summary.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mup_summary.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nSummary plot saved to {output_dir}/mup_summary.png")


def main():
    parser = argparse.ArgumentParser(description='Generate MuP verification plots')
    parser.add_argument('--output-dir', type=str, default='./mup_plots',
                        help='Directory to save plots')
    parser.add_argument('--widths', type=str, default='128,256,512,1024,2048,4096,8192',
                        help='Comma-separated list of widths to test (paper uses 128-8192)')
    parser.add_argument('--base-hidden-size', type=int, default=128,
                        help='Base hidden size for MuP')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num-seeds', type=int, default=3,
                        help='Number of random seeds for coordinate check')
    parser.add_argument('--lr-sweep-steps', type=int, default=500,
                        help='Number of steps per LR sweep run')
    parser.add_argument('--loss-curve-steps', type=int, default=500,
                        help='Number of steps for loss curve comparison')
    parser.add_argument('--skip-lr-sweep', action='store_true',
                        help='Skip LR sweep (faster)')
    parser.add_argument('--skip-loss-curves', action='store_true',
                        help='Skip loss curves (faster)')
    parser.add_argument('--skip-coord-check', action='store_true',
                        help='Skip coordinate check (run only LR sweep)')
    parser.add_argument('--coord-check-steps', type=int, default=10,
                        help='Number of training steps for per-step coordinate check')
    # Data arguments
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing preprocessed data (e.g., ./assets)')
    parser.add_argument('--use-mock-data', action='store_true',
                        help='Use MockGPTDataset instead of real data')
    parser.add_argument('--vocab-size', type=int, default=50257,
                        help='Vocabulary size (default: GPT-2 vocab size)')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length for training')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--small-vocab', action='store_true',
                        help='Use small vocab (1024) to make hidden layer params more significant')
    parser.add_argument('--debug-gradients', action='store_true',
                        help='Print per-group gradient norms during training')
    # Character-level data arguments
    parser.add_argument('--char-data', action='store_true',
                        help='Use character-level dataset (enwik8 or text8) for small vocab')
    parser.add_argument('--char-dataset', type=str, default='enwik8', choices=['enwik8', 'text8'],
                        help='Which character-level dataset to use (default: enwik8)')
    parser.add_argument('--delta-coord-check', action='store_true',
                        help='Run delta-from-init coordinate check (MuP paper Figure 8 style)')
    parser.add_argument('--delta-steps', type=int, default=4,
                        help='Number of Adam updates for delta coordinate check (default: 4, gives t=0..4)')
    parser.add_argument('--delta-seeds', type=int, default=3,
                        help='Number of random seeds for delta coordinate check std bands')
    args = parser.parse_args()

    # Parse widths
    widths = [int(w) for w in args.widths.split(',')]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MuP Verification Plots for Megatron-LM")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Output directory: {args.output_dir}")
    print(f"Widths: {widths}")
    print(f"Base hidden size: {args.base_hidden_size}")

    print("\nInitializing distributed...")
    init_distributed()
    print("Done.")

    # Initialize data provider
    print("\nSetting up data...")
    if args.char_data:
        # Use character-level dataset (enwik8 or text8) - small vocab, real data
        data_provider = CharLevelDataProvider(
            dataset_name=args.char_dataset,
            data_dir=args.data_dir or './assets',
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        print(f"Data type: Character-level ({args.char_dataset})")
        print(f"Vocab size: {data_provider.vocab_size}")
    else:
        use_mock = args.use_mock_data or args.data_dir is None
        data_provider = get_data_provider(
            data_dir=args.data_dir,
            use_mock=use_mock,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        print(f"Data type: {'Mock' if data_provider.use_mock else 'Real'}")
        print(f"Vocab size: {data_provider.vocab_size}")

    # --small-vocab only applies to non-char-data modes
    if args.small_vocab and not args.char_data:
        print(f"Note: --small-vocab is ignored when not using --char-data (use --char-data for small vocab)")

    try:
        # 1. Coordinate Check (init-time + per-step)
        if not args.skip_coord_check:
            coord_std, coord_mup, step_stats_std, step_stats_mup = run_coordinate_check(
                widths=widths,
                base_hidden_size=args.base_hidden_size,
                num_layers=args.num_layers,
                num_seeds=args.num_seeds,
                num_steps=args.coord_check_steps,
                vocab_size=data_provider.vocab_size,
                data_provider=data_provider,
            )
            plot_coordinate_check(coord_std, coord_mup, args.output_dir)

            # Plot per-step coordinate check if we have data
            if step_stats_std and any(step_stats_std.values()):
                plot_coord_check_per_step(step_stats_std, step_stats_mup, args.output_dir)

        # 1b. Delta-from-init coordinate check (MuP paper Figure 8 style)
        if args.delta_coord_check and not args.skip_coord_check:
            delta_std, delta_mup = run_delta_coordinate_check(
                widths=widths,
                base_hidden_size=args.base_hidden_size,
                num_layers=args.num_layers,
                num_steps=args.delta_steps,
                num_seeds=args.delta_seeds,
                vocab_size=data_provider.vocab_size,
                data_provider=data_provider,
            )
            plot_delta_coordinate_check(delta_std, delta_mup, args.output_dir,
                                        base_width=args.base_hidden_size)

        # 2. LR Sweep (extended range: log₂(LR) from -22 to -7)
        if not args.skip_lr_sweep:
            # Powers of 2 from 2^-22 to 2^-7 (extended range for better coverage)
            lrs = [2**i for i in range(-22, -6)]  # 2^-22 to 2^-7 (16 LRs)

            lr_std = run_lr_sweep(
                widths=widths,
                lrs=lrs,
                base_hidden_size=args.base_hidden_size,
                num_layers=args.num_layers,
                num_steps=args.lr_sweep_steps,
                use_mup=False,
                data_provider=data_provider,
                debug_gradients=args.debug_gradients,
            )

            lr_mup = run_lr_sweep(
                widths=widths,
                lrs=lrs,
                base_hidden_size=args.base_hidden_size,
                num_layers=args.num_layers,
                num_steps=args.lr_sweep_steps,
                use_mup=True,
                data_provider=data_provider,
                debug_gradients=args.debug_gradients,
            )

            plot_lr_sweep(lr_std, lr_mup, args.output_dir)
        else:
            # Use dummy data for summary plot
            lrs = [2**i for i in range(-22, -6)]
            lr_std = {w: {lr: 5.5 for lr in lrs} for w in widths}
            lr_mup = {w: {lr: 5.5 for lr in lrs} for w in widths}

        # 3. Loss Curves
        if not args.skip_loss_curves:
            loss_std, loss_mup = run_loss_curves(
                widths=widths,
                base_hidden_size=args.base_hidden_size,
                base_lr=1e-3,
                num_layers=args.num_layers,
                num_steps=args.loss_curve_steps,
                data_provider=data_provider,
                debug_gradients=args.debug_gradients,
            )
            plot_loss_curves(loss_std, loss_mup, args.output_dir)

        # 4. Summary Plot (only if we have coord check data)
        if not args.skip_coord_check:
            plot_summary(coord_std, coord_mup, lr_std, lr_mup, args.output_dir)

        # Save raw results as JSON
        if args.char_data:
            data_type = f'char-level ({args.char_dataset})'
        elif hasattr(data_provider, 'use_mock') and data_provider.use_mock:
            data_type = 'mock'
        else:
            data_type = 'real'

        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'widths': widths,
                'base_hidden_size': args.base_hidden_size,
                'num_layers': args.num_layers,
                'vocab_size': data_provider.vocab_size,
                'seq_len': args.seq_len,
                'batch_size': args.batch_size,
                'data_type': data_type,
                'data_dir': args.data_dir,
            },
        }

        # Add coordinate check results if available
        if not args.skip_coord_check:
            results['coordinate_check'] = {
                'standard': {str(k): v for k, v in coord_std.items()},
                'mup': {str(k): v for k, v in coord_mup.items()},
            }
            # Add per-step stats if available
            if step_stats_std and any(step_stats_std.values()):
                results['coordinate_check_per_step'] = {
                    'standard': {str(k): v for k, v in step_stats_std.items()},
                    'mup': {str(k): v for k, v in step_stats_mup.items()},
                }

        if not args.skip_lr_sweep:
            results['lr_sweep'] = {
                'standard': {str(k): {str(lr): v for lr, v in d.items()} for k, d in lr_std.items()},
                'mup': {str(k): {str(lr): v for lr, v in d.items()} for k, d in lr_mup.items()},
            }

        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        print(f"Plots saved to: {args.output_dir}/")
        if not args.skip_coord_check:
            print("  - coordinate_check.png")
            if step_stats_std and any(step_stats_std.values()):
                print("  - coord_check_per_step.png")
                print("  - coord_check_per_step_summary.png")
            if args.delta_coord_check:
                print("  - mup_coord_check_delta.png (MuP paper Figure 8 style)")
            print("  - mup_summary.png")
        else:
            print("  - (coordinate check skipped)")
        print("  - lr_sweep.png" if not args.skip_lr_sweep else "  - (lr_sweep skipped)")
        print("  - loss_curves.png" if not args.skip_loss_curves else "  - (loss_curves skipped)")
        print("  - results.json")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
