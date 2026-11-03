"""Task 2 (Issue R3) Verification Suite: Checkpoint Registry, Strict Loading & Codebase Hygiene."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.audit_log import AuditLog
from src.inference import (
    DEFAULT_REGISTRY_PATH,
    ModelUnavailable,
    compute_state_dict_sha256,
    load_diagnostic_backbone,
    load_model_registry,
    load_verified_checkpoint,
    run_inference,
)
from src.interlocks import load_interlock_config
from src.scan_audit import finalize_scan
from tests.test_interlocks import profile


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.pathologies = ["Pneumonia", "Pneumothorax"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class TestModelRegistryAndStrictLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_01_shipped_registry_loads_and_enforces_strict(self) -> None:
        reg = load_model_registry(DEFAULT_REGISTRY_PATH)
        self.assertIn("densenet121-res224-all", reg.entries)
        entry = reg.entries["densenet121-res224-all"]
        self.assertTrue(entry.strict_loading)
        self.assertTrue(entry.weights_only)
        self.assertEqual(len(entry.pathologies), 18)

    def test_02_missing_registry_fails_closed(self) -> None:
        with self.assertRaises(ModelUnavailable) as ctx:
            load_model_registry(self.d / "missing.yaml")
        self.assertEqual(ctx.exception.code, "REGISTRY_MISSING")

    def test_03_registry_rejects_strict_false_policy(self) -> None:
        raw = yaml.safe_load(DEFAULT_REGISTRY_PATH.read_text())
        raw["models"]["densenet121-res224-all"]["strict_loading"] = False
        p = self.d / "bad_reg.yaml"
        p.write_text(yaml.safe_dump(raw))
        with self.assertRaises(ModelUnavailable) as ctx:
            load_model_registry(p)
        self.assertEqual(ctx.exception.code, "REGISTRY_UNSAFE_POLICY")

    def test_04_deterministic_state_dict_sha256(self) -> None:
        torch.manual_seed(123)
        m1 = TinyNet()
        torch.manual_seed(123)
        m2 = TinyNet()
        h1 = compute_state_dict_sha256(m1.state_dict())
        h2 = compute_state_dict_sha256(m2.state_dict())
        self.assertEqual(h1, h2)
        with torch.no_grad():
            m2.fc.weight.add_(0.01)
        self.assertNotEqual(h1, compute_state_dict_sha256(m2.state_dict()))

    def test_05_verified_checkpoint_loads_strictly(self) -> None:
        src_net = TinyNet()
        ckpt = self.d / "tiny.pt"
        torch.save(src_net.state_dict(), ckpt)
        expected_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

        dst_net = TinyNet()
        actual_sha = load_verified_checkpoint(dst_net, ckpt, expected_sha256=expected_sha)
        self.assertEqual(actual_sha, expected_sha)
        self.assertEqual(
            compute_state_dict_sha256(src_net.state_dict()),
            compute_state_dict_sha256(dst_net.state_dict()),
        )

    def test_06_checkpoint_hash_mismatch_fails_closed(self) -> None:
        net = TinyNet()
        ckpt = self.d / "tiny.pt"
        torch.save(net.state_dict(), ckpt)
        with self.assertRaises(ModelUnavailable) as ctx:
            load_verified_checkpoint(net, ckpt, expected_sha256="0" * 64)
        self.assertEqual(ctx.exception.code, "CHECKPOINT_HASH_MISMATCH")

    def test_07_mismatched_keys_fail_strict_loading(self) -> None:
        net = TinyNet()
        ckpt = self.d / "wrong_keys.pt"
        torch.save({"unexpected_layer.weight": torch.ones(2, 4)}, ckpt)
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        with self.assertRaises(ModelUnavailable) as ctx:
            load_verified_checkpoint(net, ckpt, expected_sha256=sha)
        self.assertEqual(ctx.exception.code, "CHECKPOINT_STRICT_MISMATCH")

    def test_08_missing_checkpoint_file_raises(self) -> None:
        net = TinyNet()
        with self.assertRaises(ModelUnavailable) as ctx:
            load_verified_checkpoint(net, self.d / "nope.pt")
        self.assertEqual(ctx.exception.code, "CHECKPOINT_NOT_FOUND")

    def test_09_custom_checkpoint_requires_pinned_hash_in_registry(self) -> None:
        ckpt = self.d / "tiny.pt"
        torch.save(TinyNet().state_dict(), ckpt)
        with self.assertRaises(ModelUnavailable) as ctx:
            load_diagnostic_backbone(
                "cpu",
                checkpoint_path=ckpt,
                registry_path=DEFAULT_REGISTRY_PATH,
                model_factory=TinyNet,
            )
        self.assertEqual(ctx.exception.code, "CHECKPOINT_UNPINNED_IN_REGISTRY")

    def test_10_load_backbone_with_pinned_custom_checkpoint(self) -> None:
        net = TinyNet()
        ckpt = self.d / "tiny.pt"
        torch.save(net.state_dict(), ckpt)
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

        raw = yaml.safe_load(DEFAULT_REGISTRY_PATH.read_text())
        raw["models"]["densenet121-res224-all"]["expected_state_sha256"] = sha
        reg_p = self.d / "pinned_reg.yaml"
        reg_p.write_text(yaml.safe_dump(raw))

        loaded = load_diagnostic_backbone(
            "cpu",
            checkpoint_path=ckpt,
            registry_path=reg_p,
            model_factory=TinyNet,
        )
        self.assertEqual(loaded.model_sha256, sha)
        self.assertEqual(loaded.model_id, "densenet121-res224-all")

    def test_11_unregistered_model_id_rejected(self) -> None:
        with self.assertRaises(ModelUnavailable) as ctx:
            load_diagnostic_backbone("cpu", model_id="fabricated-net-99")
        self.assertEqual(ctx.exception.code, "MODEL_NOT_IN_REGISTRY")

    def test_12_non_finite_inference_output_rejected(self) -> None:
        class NaNModel(nn.Module):
            def forward(self, x):
                return torch.tensor([[float("nan"), 0.5]])

        with self.assertRaises(ModelUnavailable) as ctx:
            run_inference(NaNModel(), torch.zeros(1, 4), ["Pneumonia", "Edema"])
        self.assertEqual(ctx.exception.code, "NON_FINITE_PREDICTION")

    def test_13_audit_record_stores_model_sha256(self) -> None:
        img = self.d / "scan.png"
        img.write_bytes(b"img-bytes")
        sidecar = self.d / "scan_vitals.json"
        sidecar.write_text(json.dumps(profile()))
        log = AuditLog(self.d / "audit_log.jsonl")
        cfg = load_interlock_config()

        from datetime import datetime, timezone
        now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
        row = finalize_scan(
            scan_id="scan",
            img_path=img,
            run_id="r1",
            profile_path=sidecar,
            interlock_cfg=cfg,
            requested_actions=["cxr_triage"],
            thresholds={"l1_min_thoracic_pixels": 1000},
            l1={"pass": True, "thoracic_pixels": 50000},
            l2={"status": "PASSED", "aamr": 0.91},
            l3={"passed": True, "entropy": 0.25, "ambiguous_count": 0},
            predictions={"Pneumonia": 0.4},
            model_id="densenet121-res224-all",
            model_sha256="abc123def456",
            model_registry_sha256="reg999",
            audit_log=log,
            output_dir=self.d,
            now=now,
        )
        self.assertEqual(row["model_sha256"], "abc123def456")
        rec = json.loads((self.d / "audit_log.jsonl").read_text().splitlines()[-1])
        self.assertEqual(rec["model_sha256"], "abc123def456")
        self.assertEqual(rec["model_registry_sha256"], "reg999")

    def test_14_codebase_forbidden_pattern_scan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        # 1. Synthetic trainers must not exist in src/
        for forbidden_file in ("train.py", "train_cohort.py", "streaming_dataset.py"):
            self.assertFalse(
                (src_dir / forbidden_file).exists(),
                f"Unsafe trainer {forbidden_file} must be quarantined outside src/",
            )
        # 2. Scan src/*.py and main.py for strict=False or randn
        files = list(src_dir.glob("*.py")) + [repo_root / "main.py"]
        for f in files:
            text = f.read_text()
            self.assertNotIn("strict=False", text, f"Forbidden strict=False in {f.name}")
            self.assertNotIn("strict = False", text, f"Forbidden strict = False in {f.name}")
            self.assertNotIn("torch.randn", text, f"Forbidden random tensor generation in {f.name}")
            if "torch.load(" in text:
                self.assertIn("weights_only=True", text, f"torch.load without weights_only=True in {f.name}")


if __name__ == "__main__":
    unittest.main()
