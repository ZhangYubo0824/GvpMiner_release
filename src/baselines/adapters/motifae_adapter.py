#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from adapter_common import (
    build_parser,
    canonical_sequence_id,
    catalog_from_occurrences,
    find_first,
    first_value,
    read_delimited,
    write_standard_outputs,
)


def _load_id_map(input_dir: Path) -> dict[str, str]:
    mapping_path = find_first(input_dir, ["id_map.csv"])
    if mapping_path is None:
        return {}
    rows = read_delimited(mapping_path)
    return {
        str(row.get("safe_id") or row.get("uniprot") or row.get("sequence_id") or "").strip(): canonical_sequence_id(
            row.get("original_id") or row.get("protein_id") or row.get("parent_id") or ""
        )
        for row in rows
        if row.get("safe_id") or row.get("uniprot") or row.get("sequence_id")
    }


def _load_motifae_sequences(input_dir: Path) -> dict[str, str]:
    input_path = find_first(input_dir, ["motifae_input.csv"])
    if input_path is None:
        return {}
    result = {}
    for row in read_delimited(input_path):
        sequence_id = first_value(row, ["uniprot", "safe_id", "sequence_id"])
        sequence = first_value(row, ["sequence", "random_subregion", "seq"])
        if sequence_id and sequence:
            result[sequence_id] = sequence
    return result


def _activation_csv(input_dir: Path) -> Path | None:
    return find_first(
        input_dir,
        [
            "motifae_feature_activations.csv",
            "feature_activations.csv",
            "activations.csv",
            "motifae_activations.csv",
        ],
    )


def _row_feature_id(row: Mapping[str, object]) -> str:
    return first_value(row, ["feature_id", "feature", "latent_id", "neuron_id", "motif_id"])


def _row_safe_id(row: Mapping[str, object]) -> str:
    return first_value(row, ["safe_id", "uniprot", "sequence_id", "protein_id"])


def _row_score(row: Mapping[str, object]) -> float:
    value = first_value(row, ["activation", "score", "value"], "0")
    return float(value)


def _row_interval(row: Mapping[str, object]) -> tuple[int, int]:
    start = first_value(row, ["occ_start", "start", "start_0based"])
    end = first_value(row, ["occ_end", "end", "end_0based"])
    if start and end:
        return int(float(start)), int(float(end))
    start_1based = first_value(row, ["start_1based"])
    end_1based = first_value(row, ["end_1based"])
    if start_1based and end_1based:
        return max(0, int(float(start_1based)) - 1), int(float(end_1based))
    position = first_value(row, ["position", "token_index", "residue_index", "pos"])
    if not position:
        raise ValueError(f"MotifAE activation row is missing a position/start field: {row}")
    pos = int(float(position))
    return pos, pos + 1


def _merge_points(
    *,
    family: str,
    method: str,
    points: Sequence[tuple[str, str, int, int, float]],
    id_map: Mapping[str, str],
    sequences: Mapping[str, str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
    for feature_id, safe_id, start, end, score in points:
        grouped[(feature_id, safe_id)].append((start, end, score))

    occurrences = []
    for (feature_id, safe_id), intervals in sorted(grouped.items()):
        intervals.sort(key=lambda item: (item[0], item[1]))
        merged: list[tuple[int, int, float]] = []
        for start, end, score in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, end, score))
                continue
            old_start, old_end, old_score = merged[-1]
            merged[-1] = (old_start, max(old_end, end), max(old_score, score))

        sequence_id = id_map.get(safe_id) or canonical_sequence_id(safe_id)
        sequence = sequences.get(safe_id, "")
        for start, end, score in merged:
            motif_sequence = sequence[start:end] if sequence and 0 <= start < end <= len(sequence) else ""
            occurrences.append(
                {
                    "family": family,
                    "method": method,
                    "motif_uid": f"{family}_{method}_feature_{feature_id}",
                    "sequence_id": sequence_id,
                    "parent_id": sequence_id,
                    "occ_start": start,
                    "occ_end": end,
                    "score": f"{score:.6g}",
                    "significance": "",
                    "motif_sequence": motif_sequence or f"feature_{feature_id}",
                    "consensus_or_label": f"feature_{feature_id}",
                    "source_model": method,
                }
            )
    return occurrences


def _adapt_from_activation_csv(*, family: str, method: str, input_dir: Path, output_dir: Path, path: Path) -> None:
    id_map = _load_id_map(input_dir)
    sequences = _load_motifae_sequences(input_dir)
    points = []
    for row in read_delimited(path):
        feature_id = _row_feature_id(row)
        safe_id = _row_safe_id(row)
        if not feature_id or not safe_id:
            continue
        start, end = _row_interval(row)
        points.append((feature_id, safe_id, start, end, _row_score(row)))

    occurrences = _merge_points(
        family=family,
        method=method,
        points=points,
        id_map=id_map,
        sequences=sequences,
    )
    write_standard_outputs(
        output_dir=output_dir,
        method=method,
        occurrences=occurrences,
        catalog=catalog_from_occurrences(occurrences),
    )


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"step_(\d+)", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def _latest_checkpoint(input_dir: Path) -> Path | None:
    checkpoint_dir = input_dir / "sae" / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=_checkpoint_sort_key)
    return checkpoints[-1] if checkpoints else None


def _state_dict(checkpoint: object) -> Mapping[str, object]:
    if isinstance(checkpoint, Mapping):
        for key in ["model_state_dict", "state_dict", "model"]:
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint
    raise TypeError(f"Unsupported MotifAE checkpoint object: {type(checkpoint)!r}")


def _find_encoder_weights(state: Mapping[str, object], activation_dim: int):
    candidates = []
    for key, value in state.items():
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 2 or activation_dim not in tuple(shape):
            continue
        key_lower = key.lower()
        priority = 0 if any(token in key_lower for token in ["encoder", "w_enc", "encode"]) else 1
        candidates.append((priority, key, value))
    if not candidates:
        raise ValueError(
            f"Could not find MotifAE encoder weights with activation_dim={activation_dim}. "
            f"Available tensor keys: {[key for key, value in state.items() if getattr(value, 'shape', None) is not None][:20]}"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1], candidates[0][2]


def _find_encoder_bias(state: Mapping[str, object], weight_key: str, feature_dim: int, torch_module):
    candidates = []
    base = weight_key.rsplit(".", 1)[0]
    for key, value in state.items():
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) != 1 or int(shape[0]) != feature_dim:
            continue
        key_lower = key.lower()
        priority = 0 if key.startswith(base) and "bias" in key_lower else 1
        candidates.append((priority, key, value))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]
    return torch_module.zeros(feature_dim)


def _adapt_from_checkpoint(*, family: str, method: str, input_dir: Path, output_dir: Path, checkpoint_path: Path) -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MotifAE adapter found embeddings and SAE checkpoints, but PyTorch/numpy are not available. "
            "Run with the MotifAE environment, e.g. set MOTIFAE_ADAPTER_CMD to invoke this adapter with MOTIFAE_PYTHON."
        ) from exc

    id_map = _load_id_map(input_dir)
    sequences = _load_motifae_sequences(input_dir)
    embedding_dir = input_dir / "embeddings"
    embedding_paths = sorted(embedding_dir.glob("*.npz"))
    if not embedding_paths:
        raise FileNotFoundError(f"No MotifAE embedding .npz files found under {embedding_dir}")

    sample = np.load(embedding_paths[0])["repr"]
    activation_dim = int(sample.shape[-1])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _state_dict(checkpoint)
    weight_key, weight = _find_encoder_weights(state, activation_dim)
    feature_dim = int(weight.shape[0] if int(weight.shape[1]) == activation_dim else weight.shape[1])
    bias = _find_encoder_bias(state, weight_key, feature_dim, torch)
    device = os.environ.get("MOTIFAE_ADAPTER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    weight = weight.to(device)
    bias = bias.to(device)
    top_k = max(1, int(os.environ.get("MOTIFAE_TOPK_PER_TOKEN", "1")))
    top_features = max(1, int(os.environ.get("MOTIFAE_TOP_FEATURES", "200")))
    min_activation = float(os.environ.get("MOTIFAE_MIN_ACTIVATION", "0"))

    raw_points: list[tuple[str, str, int, int, float]] = []
    for embedding_path in embedding_paths:
        safe_id = embedding_path.stem
        array = np.load(embedding_path)["repr"]
        if array.ndim == 3:
            array = array[0]
        hidden = torch.as_tensor(array, dtype=weight.dtype, device=device)
        with torch.no_grad():
            if int(weight.shape[1]) == activation_dim:
                activations = torch.relu(hidden @ weight.T + bias)
            else:
                activations = torch.relu(hidden @ weight + bias)
            values, indices = torch.topk(activations, k=min(top_k, activations.shape[-1]), dim=-1)
        values = values.cpu().numpy()
        indices = indices.cpu().numpy()
        for position in range(values.shape[0]):
            for rank in range(values.shape[1]):
                score = float(values[position, rank])
                if score <= min_activation:
                    continue
                raw_points.append((str(int(indices[position, rank])), safe_id, position, position + 1, score))

    support: dict[str, int] = defaultdict(int)
    for feature_id, _safe_id, _start, _end, _score in raw_points:
        support[feature_id] += 1
    retained = {
        feature_id
        for feature_id, _count in sorted(support.items(), key=lambda item: (-item[1], item[0]))[:top_features]
    }
    points = [point for point in raw_points if point[0] in retained]
    occurrences = _merge_points(
        family=family,
        method=method,
        points=points,
        id_map=id_map,
        sequences=sequences,
    )
    write_standard_outputs(
        output_dir=output_dir,
        method=method,
        occurrences=occurrences,
        catalog=catalog_from_occurrences(occurrences),
    )


def adapt(*, family: str, method: str, input_dir: Path, output_dir: Path) -> None:
    standard_occurrence = find_first(input_dir, [f"motif_occurrences_{method}.csv", "motif_occurrences.csv"])
    standard_catalog = find_first(input_dir, [f"motif_catalog_{method}.csv", "motif_catalog.csv"])
    if standard_occurrence is not None and standard_catalog is not None:
        from standard_csv_adapter import adapt as copy_standard

        copy_standard(family=family, method=method, input_dir=input_dir, output_dir=output_dir)
        return

    activation_path = _activation_csv(input_dir)
    if activation_path is not None:
        _adapt_from_activation_csv(family=family, method=method, input_dir=input_dir, output_dir=output_dir, path=activation_path)
        return

    checkpoint = _latest_checkpoint(input_dir)
    if checkpoint is not None:
        _adapt_from_checkpoint(family=family, method=method, input_dir=input_dir, output_dir=output_dir, checkpoint_path=checkpoint)
        return

    raise FileNotFoundError(
        "MotifAE adapter needs standard motif CSVs, a feature activation CSV, or embeddings plus SAE checkpoints."
    )


def main() -> None:
    args = build_parser("Convert MotifAE feature activations to baseline motif CSV outputs.").parse_args()
    adapt(family=args.family, method=args.method, input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
