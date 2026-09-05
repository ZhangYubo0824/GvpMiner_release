from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Predictor = Callable[[str, str, Path], Path]


@dataclass
class StructureBackend:
    name: str
    predictor: Predictor | None = None

    @staticmethod
    def _pdb_output_path(output_dir: Path, sequence_id: str) -> Path:
        return Path(output_dir) / f"{sequence_id}.pdb"

    def predict(self, sequence_id: str, sequence: str, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._pdb_output_path(output_dir, sequence_id)
        if output_path.exists():
            return output_path
        if self.predictor is None:
            raise RuntimeError(f"Backend '{self.name}' has no local predictor configured.")
        return Path(self.predictor(sequence_id, sequence, output_dir))


class ESMFoldBackend(StructureBackend):
    _model = None
    _device = None
    _fold_model_name = "esmfold_3B_v1"
    _language_model_name = "esm2_t36_3B_UR50D"
    _runtime_dependencies = (
        "torch",
        "esm",
        "omegaconf",
        "fairscale",
        "pytorch_lightning",
        "biotite",
        "openfold",
    )

    def __init__(self, predictor: Predictor | None = None):
        super().__init__(name="esmfold", predictor=predictor)

    @classmethod
    def _project_root(cls) -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _models_dir(cls) -> Path:
        return cls._project_root() / "models"

    @classmethod
    def _model_path(cls, model_name: str) -> Path:
        return cls._models_dir() / f"{model_name}.pt"

    @classmethod
    def _prepare_model_cache(cls, torch_module) -> None:
        models_dir = cls._models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        torch_module.hub.set_dir(str(cls._project_root()))

    @classmethod
    def _patch_torch_download_dir(cls, torch_module):
        original_download = torch_module.hub.load_state_dict_from_url

        def load_state_dict_from_url_to_models(url, *args, **kwargs):
            kwargs.setdefault("model_dir", str(cls._models_dir()))
            return original_download(url, *args, **kwargs)

        torch_module.hub.load_state_dict_from_url = load_state_dict_from_url_to_models
        return original_download

    @classmethod
    def _local_or_hub_model_name(cls, model_name: str) -> str:
        model_path = cls._model_path(model_name)
        return str(model_path) if model_path.exists() else model_name

    @classmethod
    def _load_esmfold_model(cls, model_name: str):
        pretrained = importlib.import_module("esm.esmfold.v1.pretrained")
        return pretrained._load_model(model_name)

    @classmethod
    def _patch_esmfold_language_model_loader(cls):
        import esm

        original_loader = esm.pretrained.esm2_t36_3B_UR50D

        def load_local_or_hub_language_model():
            return esm.pretrained.load_model_and_alphabet(
                cls._local_or_hub_model_name(cls._language_model_name)
            )

        esm.pretrained.esm2_t36_3B_UR50D = load_local_or_hub_language_model
        return original_loader

    @classmethod
    def _candidate_dependency_roots(cls, package_name: str) -> list[Path]:
        workspace_root = Path(__file__).resolve().parents[1]
        candidates: list[Path] = []
        if package_name == "openfold":
            for env_key in ("MOTIF_STRUCTURE_OPENFOLD_ROOT", "EXP5_OPENFOLD_ROOT", "OPENFOLD_HOME"):
                env_value = os.environ.get(env_key)
                if env_value:
                    candidates.append(Path(env_value))
            candidates.extend([workspace_root / "openfold_esmfold", workspace_root / "openfold"])
        else:
            candidates.append(workspace_root / package_name)
        return candidates

    @classmethod
    def _find_local_dependency_root(cls, package_name: str) -> Path | None:
        for candidate_root in cls._candidate_dependency_roots(package_name):
            package_dir = candidate_root / package_name
            if package_dir.is_dir() and (package_dir / "__init__.py").exists():
                return candidate_root
        return None

    @classmethod
    def _ensure_local_dependency_on_path(cls, package_name: str) -> None:
        candidate_root = cls._find_local_dependency_root(package_name)
        if candidate_root is not None and str(candidate_root) not in sys.path:
            sys.path.insert(0, str(candidate_root))

    @classmethod
    def missing_runtime_dependencies(cls) -> list[str]:
        missing = []
        for package_name in cls._runtime_dependencies:
            if package_name == "openfold":
                if importlib.util.find_spec("openfold") is None and cls._find_local_dependency_root("openfold") is None:
                    missing.append(package_name)
                continue
            if importlib.util.find_spec(package_name) is None:
                missing.append(package_name)
        return sorted(missing)

    @classmethod
    def _ensure_openfold_attention_core_fallback(cls) -> None:
        if "attn_core_inplace_cuda" in sys.modules:
            return
        try:
            __import__("attn_core_inplace_cuda")
            return
        except ModuleNotFoundError:
            pass
        import torch

        fallback = types.ModuleType("attn_core_inplace_cuda")

        def forward_(attention_logits, rows, cols):
            attention_logits.copy_(torch.softmax(attention_logits, dim=-1))

        def backward_(attention_logits, grad_output, v, rows, cols, _value_dim):
            grad_attention = torch.matmul(grad_output, v.transpose(-1, -2))
            grad_logits = attention_logits * (
                grad_attention - torch.sum(grad_attention * attention_logits, dim=-1, keepdim=True)
            )
            attention_logits.copy_(grad_logits)

        fallback.forward_ = forward_
        fallback.backward_ = backward_
        sys.modules["attn_core_inplace_cuda"] = fallback

    @classmethod
    def _load_model(cls):
        if cls._model is not None:
            return cls._model
        try:
            import esm
            import torch
        except ImportError as exc:
            raise RuntimeError("ESMFold requires local torch and fair-esm packages.") from exc
        cls._prepare_model_cache(torch)
        try:
            cls._ensure_local_dependency_on_path("openfold")
            cls._ensure_openfold_attention_core_fallback()
            original_language_model_loader = cls._patch_esmfold_language_model_loader()
            original_download = cls._patch_torch_download_dir(torch)
            try:
                model = cls._load_esmfold_model(
                    cls._local_or_hub_model_name(cls._fold_model_name)
                )
            finally:
                torch.hub.load_state_dict_from_url = original_download
                esm.pretrained.esm2_t36_3B_UR50D = original_language_model_loader
        except ModuleNotFoundError as exc:
            missing_module = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(f"Failed to initialize ESMFold because module is missing: {missing_module}") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.eval()
        if device == "cuda":
            model = model.cuda()
        try:
            from config import EXP4_ESMFOLD_CHUNK_SIZE

            model.set_chunk_size(EXP4_ESMFOLD_CHUNK_SIZE)
        except Exception:
            pass
        cls._model = model
        cls._device = device
        return cls._model

    def predict(self, sequence_id: str, sequence: str, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._pdb_output_path(output_dir, sequence_id)
        if output_path.exists():
            return output_path
        if self.predictor is not None:
            return super().predict(sequence_id, sequence, output_dir)
        model = self._load_model()
        try:
            import torch

            with torch.no_grad():
                pdb_text = model.infer_pdb(sequence)
        except Exception as exc:
            raise RuntimeError(f"Local ESMFold inference failed: {exc}") from exc
        output_path.write_text(pdb_text, encoding="utf-8")
        return output_path


def get_structure_backend(name: str, predictor: Predictor | None = None) -> StructureBackend:
    key = str(name).strip().lower()
    if key == "esmfold":
        return ESMFoldBackend(predictor=predictor)
    raise ValueError(f"Unknown structure backend: {name}")
