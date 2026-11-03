"""Diagnostic Backbone Loader & Inference Engine (Issue R3 compliant).

Safety invariants (Section 4 Rules 1, 2, 3, 8):
- No silent fallbacks: if a requested checkpoint is missing, unreadable, or fails
  SHA-256 / state_dict verification, raise ModelUnavailable immediately.
- Never disable strict mode during load_state_dict (always strict=True).
- Never call torch.load without weights_only=True.
- Every loaded model computes and exposes its deterministic weight SHA-256 hash
  and registry provenance for inclusion in the hash-chained audit log.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import yaml

from src.tier0_thorax_router import resize_image, to_xrv_range

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "config" / "model_registry.yaml"
DEFAULT_MODEL_ID = "densenet121-res224-all"
INPUT_SIZE = 224


class ModelUnavailable(Exception):
    """Raised when a model or checkpoint is missing, unverified, or fails strict loading."""

    def __init__(self, code: str, message: str, *, model_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.model_id = model_id


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    architecture: str
    weights_id: str
    input_resolution: Tuple[int, int]
    strict_loading: bool
    weights_only: bool
    expected_state_sha256: Optional[str]
    allow_official_xrv_pretrained: bool
    training_cohorts: Tuple[str, ...]
    validation_status: str
    pathologies: Tuple[str, ...]


@dataclass(frozen=True)
class LoadedModelRegistry:
    path: Path
    sha256: str
    entries: Dict[str, ModelRegistryEntry]


def load_model_registry(path: Optional[Union[str, Path]] = None) -> LoadedModelRegistry:
    p = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not p.is_file():
        raise ModelUnavailable("REGISTRY_MISSING", f"Model registry file not found: {p}")
    try:
        raw = p.read_bytes()
        data = yaml.safe_load(raw)
    except Exception as exc:
        raise ModelUnavailable("REGISTRY_UNREADABLE", f"Cannot parse model registry {p}: {exc}") from exc

    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ModelUnavailable("REGISTRY_SCHEMA_INVALID", "model_registry.yaml schema_version must be 1")

    models_raw = data.get("models")
    if not isinstance(models_raw, dict) or not models_raw:
        raise ModelUnavailable("REGISTRY_EMPTY", "model_registry.yaml contains no models")

    entries: Dict[str, ModelRegistryEntry] = {}
    for mid, spec in models_raw.items():
        if not isinstance(spec, dict):
            raise ModelUnavailable("REGISTRY_ENTRY_INVALID", f"Invalid entry for {mid!r}")
        if spec.get("strict_loading") is not True or spec.get("weights_only") is not True:
            raise ModelUnavailable(
                "REGISTRY_UNSAFE_POLICY",
                f"Model {mid!r} must enforce strict_loading=true and weights_only=true",
                model_id=mid,
            )
        res = spec.get("input_resolution", [224, 224])
        pathologies = tuple(str(x) for x in spec.get("pathologies", ()))
        if not pathologies:
            raise ModelUnavailable("REGISTRY_NO_PATHOLOGIES", f"Model {mid!r} lists no pathologies", model_id=mid)

        entries[str(mid)] = ModelRegistryEntry(
            model_id=str(mid),
            architecture=str(spec.get("architecture", "")),
            weights_id=str(spec.get("weights_id", mid)),
            input_resolution=(int(res[0]), int(res[1])),
            strict_loading=True,
            weights_only=True,
            expected_state_sha256=spec.get("expected_state_sha256"),
            allow_official_xrv_pretrained=bool(spec.get("allow_official_xrv_pretrained", False)),
            training_cohorts=tuple(str(c) for c in spec.get("training_cohorts", ())),
            validation_status=str(spec.get("validation_status", "UNVALIDATED")),
            pathologies=pathologies,
        )

    return LoadedModelRegistry(
        path=p,
        sha256=hashlib.sha256(raw).hexdigest(),
        entries=entries,
    )


def compute_state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 over parameter names, shapes, dtypes, and bytes."""
    if not state_dict:
        raise ModelUnavailable("EMPTY_STATE_DICT", "Cannot hash an empty state_dict")
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise ModelUnavailable("NON_TENSOR_WEIGHT", f"Weight entry {key!r} is not a torch.Tensor")
        cpu_t = tensor.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(cpu_t.shape)).encode("utf-8"))
        h.update(str(cpu_t.dtype).encode("utf-8"))
        h.update(cpu_t.numpy().tobytes())
    return h.hexdigest()


def load_verified_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    *,
    expected_sha256: Optional[str] = None,
    model_id: Optional[str] = None,
) -> str:
    """Load a local checkpoint with weights_only=True and strict=True. Never falls back."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise ModelUnavailable(
            "CHECKPOINT_NOT_FOUND",
            f"Checkpoint file does not exist: {p}",
            model_id=model_id,
        )

    file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    if expected_sha256 is not None and file_hash.lower() != expected_sha256.strip().lower():
        raise ModelUnavailable(
            "CHECKPOINT_HASH_MISMATCH",
            f"Checkpoint {p.name} SHA-256 {file_hash} != expected {expected_sha256}",
            model_id=model_id,
        )

    try:
        state = torch.load(str(p), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelUnavailable(
            "CHECKPOINT_LOAD_FAILED",
            f"torch.load(weights_only=True) rejected {p.name}: {exc}",
            model_id=model_id,
        ) from exc

    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]

    if not isinstance(state, Mapping):
        raise ModelUnavailable(
            "CHECKPOINT_INVALID_FORMAT",
            f"Checkpoint {p.name} did not contain a state_dict mapping",
            model_id=model_id,
        )

    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ModelUnavailable(
            "CHECKPOINT_STRICT_MISMATCH",
            f"load_state_dict(strict=True) failed for {p.name}: {exc}",
            model_id=model_id,
        ) from exc

    return file_hash


def load_diagnostic_backbone(
    device: Union[str, torch.device] = "cpu",
    *,
    model_id: str = DEFAULT_MODEL_ID,
    checkpoint_path: Optional[Union[str, Path]] = None,
    registry_path: Optional[Union[str, Path]] = None,
    model_factory: Optional[Any] = None,
) -> Any:
    """Load and verify the diagnostic backbone against config/model_registry.yaml.

    Attaches `.model_id`, `.model_sha256`, and `.registry_sha256` to the returned model.
    Never silently falls back on error.
    """
    reg = load_model_registry(registry_path)
    if model_id not in reg.entries:
        raise ModelUnavailable(
            "MODEL_NOT_IN_REGISTRY",
            f"Model ID {model_id!r} is not registered in {reg.path.name}",
            model_id=model_id,
        )
    entry = reg.entries[model_id]
    dev = torch.device(device)

    print(f"[BACKBONE] Initializing {entry.model_id} on {dev} (status={entry.validation_status})...")

    if checkpoint_path is not None:
        # Custom checkpoint path supplied: build uninitialized architecture or use factory,
        # then require verified checkpoint with strict=True and weights_only=True.
        if entry.expected_state_sha256 is None:
            raise ModelUnavailable(
                "CHECKPOINT_UNPINNED_IN_REGISTRY",
                f"Custom checkpoint supplied for {model_id!r}, but expected_state_sha256 is null in registry.",
                model_id=model_id,
            )
        try:
            if model_factory is not None:
                model = model_factory()
            else:
                import torchxrayvision as xrv
                model = xrv.models.DenseNet(weights=None)
        except Exception as exc:
            raise ModelUnavailable(
                "ARCHITECTURE_INIT_FAILED",
                f"Failed to instantiate architecture for {model_id!r}: {exc}",
                model_id=model_id,
            ) from exc

        weight_sha256 = load_verified_checkpoint(
            model,
            checkpoint_path,
            expected_sha256=entry.expected_state_sha256,
            model_id=model_id,
        )
    else:
        if not entry.allow_official_xrv_pretrained:
            raise ModelUnavailable(
                "PRETRAINED_DISABLED",
                f"Model {model_id!r} requires an explicit verified checkpoint_path.",
                model_id=model_id,
            )
        try:
            if model_factory is not None:
                model = model_factory()
            else:
                import torchxrayvision as xrv
                model = xrv.models.DenseNet(weights=entry.weights_id)
        except Exception as exc:
            raise ModelUnavailable(
                "BACKBONE_LOAD_FAILED",
                f"Failed to load official backbone {entry.weights_id!r}: {exc}",
                model_id=model_id,
            ) from exc

        weight_sha256 = compute_state_dict_sha256(model.state_dict())
        if (
            entry.expected_state_sha256 is not None
            and weight_sha256.lower() != entry.expected_state_sha256.strip().lower()
        ):
            raise ModelUnavailable(
                "STATE_DICT_HASH_MISMATCH",
                f"Loaded state_dict SHA-256 {weight_sha256} != registry {entry.expected_state_sha256}",
                model_id=model_id,
            )

    model = model.to(dev)
    model.eval()
    model.model_id = entry.model_id
    model.model_sha256 = weight_sha256
    model.registry_sha256 = reg.sha256
    model.validation_status = entry.validation_status
    print(f"[BACKBONE] Verified {entry.model_id} | weights_sha256={weight_sha256[:12]}...")
    return model


def preprocess_xray(
    img_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load a 2D radiograph, resize to 224x224, and scale to [-1024, 1024]."""
    p = Path(img_path)
    if not p.is_file():
        raise FileNotFoundError(f"Radiograph not found: {p}")
    raw_pil = Image.open(p).convert("L")
    raw_arr = np.asarray(raw_pil, dtype=np.float32)
    resized = resize_image(raw_arr, INPUT_SIZE)
    xrv_arr = to_xrv_range(resized)
    tensor = torch.from_numpy(xrv_arr)[None, None].to(device=torch.device(device), dtype=torch.float32)
    return tensor, resized


def run_inference(
    model: Any,
    input_tensor: torch.Tensor,
    pathologies: Sequence[str],
) -> Dict[str, float]:
    """Run forward pass and return calibrated pathology dictionary (no fabricated values)."""
    if model is None:
        raise ModelUnavailable("MODEL_NONE", "Cannot run inference with model=None")
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = outputs.detach().cpu().numpy()[0]

    if not np.all(np.isfinite(probs)):
        raise ModelUnavailable("NON_FINITE_PREDICTION", "Model emitted NaN or Inf probabilities")

    result: Dict[str, float] = {}
    for idx, name in enumerate(pathologies):
        if name and str(name).strip():
            result[str(name)] = float(probs[idx])
    return result
