#!/usr/bin/env python3
"""Convert PersonaPlex safetensors to GGUF Q8_0 format for moshi Rust backend.

This script:
1. Loads PersonaPlex safetensors weights
2. Remaps weight names to match moshi's standard Depformer layout
3. Splits concatenated per-slice attention weights
4. Quantizes all linear weights to Q8_0
5. Writes a GGUF file compatible with candle's quantized_var_builder

Usage:
    python3 convert_personaplex_gguf.py \
        --input ~/.cache/huggingface/personaplex-7b-v1/model.safetensors \
        --output ~/.cache/huggingface/personaplex-7b-v1/model.q8.gguf \
        --num-slices 8
"""

import argparse
import numpy as np
from pathlib import Path
from safetensors import safe_open
import struct
import sys
import torch

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" as bytes: G=0x47, G=0x47, U=0x55, F=0x46
GGUF_VERSION = 2

# GGML types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30

# Q8_0 block size
Q8_0_BLOCK_SIZE = 32


def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize a float array to Q8_0 format.
    
    Q8_0: each block of 32 values gets one f16 scale + 32 int8 values.
    """
    data = data.astype(np.float32).flatten()
    n = len(data)
    # Pad to multiple of block size
    if n % Q8_0_BLOCK_SIZE != 0:
        pad = Q8_0_BLOCK_SIZE - (n % Q8_0_BLOCK_SIZE)
        data = np.concatenate([data, np.zeros(pad, dtype=np.float32)])
        n = len(data)
    
    n_blocks = n // Q8_0_BLOCK_SIZE
    blocks = data.reshape(n_blocks, Q8_0_BLOCK_SIZE)
    
    result = bytearray()
    for block in blocks:
        # Scale = max absolute value / 127
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 0.0
        
        # Quantize
        if scale > 0:
            quantized = np.round(block / scale).astype(np.int8)
        else:
            quantized = np.zeros(Q8_0_BLOCK_SIZE, dtype=np.int8)
        
        # Write scale as f16 + quantized values as int8
        result.extend(struct.pack('<e', np.float16(scale)))  # f16 scale
        result.extend(quantized.tobytes())  # 32 x int8
    
    return bytes(result)


def write_gguf_string(f, s: str):
    """Write a GGUF string (u64 length + utf8 bytes, no null terminator)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def main():
    parser = argparse.ArgumentParser(description='Convert PersonaPlex to GGUF Q8_0')
    parser.add_argument('--input', required=True, help='Input safetensors file')
    parser.add_argument('--output', required=True, help='Output GGUF file')
    parser.add_argument('--num-slices', type=int, default=8, help='Number of Depformer slices (8 or 16)')
    args = parser.parse_args()

    num_slices = args.num_slices
    num_layers = 6
    d_model = 1024

    print(f"Loading {args.input}...")
    with safe_open(args.input, framework='pt') as f:
        keys = list(f.keys())
        print(f"  {len(keys)} tensors")

        # Build remapped tensors
        remapped = {}

        def to_np(t):
            """Convert torch tensor (possibly BF16) to numpy float32."""
            return t.to(torch.float32).numpy()

        for name in keys:
            tensor = f.get_tensor(name)
            
            # Skip depformer and linears - we'll remap them
            if name.startswith('depformer') or name.startswith('linears.'):
                continue
            
            remapped[name] = to_np(tensor)

        print(f"Remapping Depformer weights for {num_slices} slices...")

        for slice_idx in range(num_slices):
            prefix = f"depformer.{slice_idx}"

            for layer_idx in range(num_layers):
                src_prefix = f"depformer.layers.{layer_idx}"
                dst_prefix = f"{prefix}.transformer.layers.{layer_idx}"

                # Self-attention in_proj: narrow from [16*3072, 1024] to [3072, 1024]
                in_proj_key = f"{src_prefix}.self_attn.in_proj_weight"
                t = to_np(f.get_tensor(in_proj_key))
                slice_w = t[slice_idx * 3 * d_model : (slice_idx + 1) * 3 * d_model]
                remapped[f"{dst_prefix}.self_attn.in_proj_weight"] = np.ascontiguousarray(slice_w)

                # Self-attention out_proj
                out_proj_key = f"{src_prefix}.self_attn.out_proj.weight"
                t = to_np(f.get_tensor(out_proj_key))
                slice_w = t[slice_idx * d_model : (slice_idx + 1) * d_model]
                remapped[f"{dst_prefix}.self_attn.out_proj.weight"] = np.ascontiguousarray(slice_w)

                # Shared norms
                for norm_name in ["norm1.alpha", "norm2.alpha"]:
                    src_key = f"{src_prefix}.{norm_name}"
                    t = to_np(f.get_tensor(src_key))
                    remapped[f"{dst_prefix}.{norm_name}"] = t

                # Per-slice gating
                for suffix in ["linear_in.weight", "linear_out.weight"]:
                    src_key = f"{src_prefix}.gating.{slice_idx}.{suffix}"
                    t = to_np(f.get_tensor(src_key))
                    remapped[f"{dst_prefix}.gating.{suffix}"] = np.ascontiguousarray(t)

            # Embeddings
            if slice_idx == 0:
                t = to_np(f.get_tensor("depformer_text_emb.weight"))
                remapped[f"{prefix}.emb.weight"] = t
            else:
                t = to_np(f.get_tensor(f"depformer_emb.{slice_idx - 1}.weight"))
                remapped[f"{prefix}.emb.weight"] = t

            # linear_in
            t = to_np(f.get_tensor(f"depformer_in.{slice_idx}.weight"))
            remapped[f"{prefix}.linear_in.weight"] = t

            # linear_out
            t = to_np(f.get_tensor(f"linears.{slice_idx}.weight"))
            remapped[f"{prefix}.linear_out.weight"] = t

    print(f"  {len(remapped)} remapped tensors")

    # Decide which tensors to quantize (2D weight matrices only, not norms/embeddings)
    quantize_tensors = {}
    keep_f32_tensors = {}
    
    for name, tensor in remapped.items():
        if tensor.ndim == 2 and 'emb' not in name and 'alpha' not in name:
            quantize_tensors[name] = tensor
        else:
            keep_f32_tensors[name] = tensor

    print(f"  {len(quantize_tensors)} tensors to quantize to Q8_0")
    print(f"  {len(keep_f32_tensors)} tensors to keep as F32")

    # Write GGUF file
    print(f"\nWriting {args.output}...")
    
    total_tensors = len(quantize_tensors) + len(keep_f32_tensors)
    
    with open(args.output, 'wb') as out:
        # Header
        out.write(struct.pack('<I', GGUF_MAGIC))
        out.write(struct.pack('<I', GGUF_VERSION))
        out.write(struct.pack('<Q', total_tensors))  # n_tensors
        out.write(struct.pack('<Q', 0))  # n_kv (no metadata)

        # Tensor info section
        tensor_data_list = []  # (name, data_bytes, ggml_type, shape)
        
        # Prepare all tensor data first to compute offsets
        all_tensors = []
        for name, tensor in {**keep_f32_tensors, **quantize_tensors}.items():
            if name in quantize_tensors:
                data = quantize_q8_0(tensor)
                ggml_type = GGML_TYPE_Q8_0
            else:
                data = tensor.astype(np.float32).tobytes()
                ggml_type = GGML_TYPE_F32
            all_tensors.append((name, data, ggml_type, tensor.shape))

        # Write tensor infos
        # GGUF stores dimensions in reverse order (column-major convention)
        offset = 0
        for name, data, ggml_type, shape in all_tensors:
            write_gguf_string(out, name)
            out.write(struct.pack('<I', len(shape)))  # n_dims
            for dim in reversed(shape):  # GGUF uses reversed dim order
                out.write(struct.pack('<Q', dim))
            out.write(struct.pack('<I', ggml_type))
            out.write(struct.pack('<Q', offset))  # offset
            # Align offset to 32 bytes
            data_len = len(data)
            offset += data_len
            padding = (32 - (data_len % 32)) % 32
            offset += padding

        # Alignment padding before tensor data
        current_pos = out.tell()
        align_padding = (32 - (current_pos % 32)) % 32
        out.write(b'\x00' * align_padding)

        # Write tensor data
        for i, (name, data, ggml_type, shape) in enumerate(all_tensors):
            out.write(data)
            # Pad to 32-byte alignment
            padding = (32 - (len(data) % 32)) % 32
            out.write(b'\x00' * padding)
            
            if (i + 1) % 50 == 0:
                print(f"  Written {i+1}/{len(all_tensors)} tensors...")

    output_size = Path(args.output).stat().st_size / 1e9
    print(f"\nDone! Output: {args.output} ({output_size:.2f} GB)")
    print(f"Reduction: original BF16 ~15GB -> Q8_0 ~{output_size:.1f}GB")


if __name__ == '__main__':
    main()
