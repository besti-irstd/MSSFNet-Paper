"""Compare the paper-named copy with the unmodified original implementation.

Run from any directory with the Python environment that provides PyTorch:
    python tools/verify_refactor.py --original-root ../MSSFNet-Code

The selective scan below is a TEST-ONLY, float32, real-valued mathematical
reference. It is injected into both implementations only in this process.
It is not a runtime fallback and does not validate the official CUDA kernel,
GPU training, dataset metrics, or numerical parity between CPU and CUDA.
The original tree is read-only; bytecode generation is disabled before imports.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

import torch
import torch.nn.functional as F


def expected_paper_path(name):
    """Independent specification of the documented legacy-to-paper mapping."""
    name = re.sub(r"^(?:module\.)+", "", name)
    if name == "module":
        return ""
    rules = (
        (r"^dwt_hpf(?=\.|$)", "awfe"),
        (r"^freq_down([123])(?=\.|$)", r"hwfp.stage\1"),
        (r"^fusion_(e3|d0)(?=\.|$)", r"fcsm_\1"),
        (r"(^|\.)ca(?=\.|$)", r"\1csa.channel_attention"),
        (r"(^|\.)sa(?=\.|$)", r"\1csa.spatial_attention"),
    )
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


def selective_scan_reference(u, delta, A, B, C, D=None, z=None,
                             delta_bias=None, delta_softplus=False,
                             return_last_state=False):
    """Test-only selective state-space recurrence for the shape used by FCSM.

    h[t] = exp(delta[t] * A) * h[t-1] + delta[t] * B[t] * u[t]
    y[t] = sum(C[t] * h[t]) + D * u[t]
    """
    if A.is_complex() or B.ndim != 3 or C.ndim != 3:
        raise ValueError("This test reference supports real, variable B/C only.")
    dtype = u.dtype
    u, delta = u.float(), delta.float()
    A, B, C = A.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    transition = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    drive = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    steps = []
    for index in range(u.shape[-1]):
        state = transition[:, :, index] * state + drive[:, :, index]
        steps.append(torch.einsum("bdn,bn->bd", state, C[:, :, index]))
    output = torch.stack(steps, dim=-1)
    if D is not None:
        output = output + u * D.float()[None, :, None]
    if z is not None:
        output = output * F.silu(z)
    output = output.to(dtype)
    return (output, state) if return_last_state else output


def compare_tensors(left, right):
    assert left.shape == right.shape, (tuple(left.shape), tuple(right.shape))
    assert left.dtype == right.dtype, (left.dtype, right.dtype)
    assert torch.isfinite(left).all().item(), "Original tensor is non-finite."
    assert torch.isfinite(right).all().item(), "Paper tensor is non-finite."
    difference = float((left - right).abs().max().item()) if left.numel() else 0.0
    assert torch.equal(left, right), f"Tensors differ; max absolute error = {difference}"
    return {"shape": list(left.shape), "max_absolute_error": difference,
            "exactly_equal": True}


def compare_state_dicts(old_state, new_state):
    mapping = {name: expected_paper_path(name) for name in old_state}
    assert len(set(mapping.values())) == len(mapping), "Mapping is not one-to-one."
    assert set(mapping.values()) == set(new_state), {
        "missing": sorted(set(mapping.values()) - set(new_state)),
        "unexpected": sorted(set(new_state) - set(mapping.values())),
    }
    for name, renamed in mapping.items():
        compare_tensors(old_state[name], new_state[renamed])
    return {"registered_tensor_count": len(mapping), "exactly_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path,
                        default=PAPER_ROOT.parent / "MSSFNet-Code")
    parser.add_argument("--report", type=Path,
                        default=PAPER_ROOT / "verification" / "refactor_report.json")
    args = parser.parse_args()
    original_root = args.original_root.resolve()
    report_path = args.report.resolve()
    if report_path == original_root or original_root in report_path.parents:
        raise ValueError("Verification output must not be inside the original tree.")
    torch.set_num_threads(1)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_root": str(original_root),
        "paper_root": str(PAPER_ROOT),
        "environment": {"python": platform.python_version(),
                        "executable": sys.executable, "torch": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "comparison_device": "cpu", "bytecode_disabled": True},
        "limits": [
            "Full-network comparisons use the same test-only real-valued selective scan reference in both models.",
            "These checks do not validate the official Mamba CUDA kernel, GPU training, or dataset accuracy.",
            "Reference injection is local to this verification process and is not a product fallback.",
        ],
        "checks": [],
    }

    def check(name, operation):
        started = time.perf_counter()
        entry = {"name": name}
        try:
            entry["details"] = operation()
            entry["status"] = "pass"
        except Exception as error:
            entry.update(status="fail", error=f"{type(error).__name__}: {error}",
                         traceback=traceback.format_exc())
        entry["seconds"] = round(time.perf_counter() - started, 4)
        report["checks"].append(entry)
        print(f"{entry['status'].upper()}: {name}", flush=True)
        if entry["status"] == "fail":
            print(entry["error"], flush=True)
        return entry["status"] == "pass"

    try:
        previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        spec = importlib.util.spec_from_file_location(
            "original_mssfnet_for_verification", original_root / "model" / "MSSFNet.py")
        original_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(original_module)
        finally:
            if previous_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices
        from model.MSSFNet import MSSFNet
        from utils.checkpoint import extract_state_dict, load_checkpoint, remap_state_dict
        fcsm_module = importlib.import_module("model.modules.fcsm")
        report["environment"]["official_selective_scan_importable"] = (
            fcsm_module.selective_scan_fn is not None)

        torch.manual_seed(20261012)
        original = original_module.HDNet(3).cpu().eval()
        torch.manual_seed(20261012)
        paper = MSSFNet(3).cpu().eval()

        def structure():
            old_parameters = dict(original.named_parameters())
            new_parameters = dict(paper.named_parameters())
            old_buffers = dict(original.named_buffers())
            new_buffers = dict(paper.named_buffers())
            for old_items, new_items in ((old_parameters, new_parameters),
                                          (old_buffers, new_buffers)):
                assert {expected_paper_path(k) for k in old_items} == set(new_items)
                assert [expected_paper_path(k) for k in old_items] == list(new_items), "Registration order changed."
                for name, tensor in old_items.items():
                    counterpart = new_items[expected_paper_path(name)]
                    assert tensor.shape == counterpart.shape, name
                    assert tensor.requires_grad == counterpart.requires_grad, name
            counts = lambda model: {
                "all_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "registered_parameter_tensors": len(list(model.parameters())),
                "registered_buffer_tensors": len(list(model.buffers())),
                "state_dict_tensors": len(model.state_dict()),
            }
            assert counts(original) == counts(paper)
            return counts(paper)

        check("parameter_buffer_registration_and_counts", structure)
        check("same_seed_initialization_exact_equality",
              lambda: compare_state_dicts(original.state_dict(), paper.state_dict()))

        def metadata():
            old = original.state_dict()
            mapped = remap_state_dict(old)
            assert list(mapped) == [expected_paper_path(k) for k in old]
            assert len(mapped._metadata) == len(old._metadata)
            for key, value in old._metadata.items():
                assert mapped._metadata[expected_paper_path(key)] == value, key
            assert all(mapped[expected_paper_path(k)] is tensor for k, tensor in old.items())
            return {"preserved_metadata_entries": len(mapped._metadata),
                    "tensor_objects_unchanged": True}

        check("state_dict_mapping_and_version_metadata", metadata)
        successful_weight = None
        for dataset in ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"):
            for filename in ("weight.pkl", "checkpoint.pkl"):
                source = original_root / "weight" / dataset / filename

                def load_actual(source=source):
                    payload = load_checkpoint(source, map_location="cpu")
                    old_state = extract_state_dict(payload)
                    old_result = original.load_state_dict(old_state, strict=True)
                    new_result = paper.load_state_dict(payload, strict=True)
                    assert not old_result.missing_keys and not old_result.unexpected_keys
                    assert not new_result.missing_keys and not new_result.unexpected_keys
                    result = compare_state_dicts(original.state_dict(), paper.state_dict())
                    result["source"] = str(source)
                    return result

                if check(f"strict_load_{dataset}_{filename}", load_actual):
                    if successful_weight is None and filename == "weight.pkl":
                        successful_weight = source

        def prefix_wrappers():
            old_state = original.state_dict()
            prefixed = OrderedDict(("module." + key, value) for key, value in old_state.items())
            prefixed._metadata = OrderedDict([("", {"version": 1})])
            prefixed._metadata.update(
                ("module" + ("." + key if key else ""), value)
                for key, value in old_state._metadata.items())
            mapped = remap_state_dict(prefixed)
            assert mapped._metadata == remap_state_dict(old_state)._metadata
            double_wrapped = torch.nn.DataParallel(torch.nn.DataParallel(original)).state_dict()
            assert remap_state_dict(double_wrapped)._metadata == remap_state_dict(old_state)._metadata
            cases = (prefixed, {"net": prefixed}, {"state_dict": prefixed},
                     {"net": {"state_dict": prefixed}}, double_wrapped,
                     {"net": double_wrapped})
            for payload in cases:
                result = paper.load_state_dict(payload, strict=True)
                assert not result.missing_keys and not result.unexpected_keys
                compare_state_dicts(old_state, paper.state_dict())
            return {"strict_load_cases": len(cases), "metadata_preserved": True}

        check("data_parallel_prefix_and_checkpoint_wrappers", prefix_wrappers)

        def save_reload():
            state = paper.state_dict()
            data = io.BytesIO()
            torch.save(state, data)
            data.seek(0)
            restored = torch.load(data, map_location="cpu", weights_only=True)
            fresh = MSSFNet(3).eval()
            fresh.load_state_dict(restored, strict=True)
            for key, value in state.items():
                compare_tensors(value, fresh.state_dict()[key])
            assert restored._metadata == state._metadata
            return {"tensor_count": len(state), "storage": "in-memory serialized checkpoint",
                    "metadata_preserved": True}

        check("paper_named_weights_save_and_reload", save_reload)

        if successful_weight is None:
            raise RuntimeError("No original weight.pkl passed strict loading; no forward test is valid.")
        payload = load_checkpoint(successful_weight)
        original.load_state_dict(extract_state_dict(payload), strict=True)
        paper.load_state_dict(payload, strict=True)
        report["forward_weight_source"] = str(successful_weight)
        torch.manual_seed(98765)
        image = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            check("AWFE_output", lambda: compare_tensors(original.dwt_hpf(image), paper.awfe(image)))

            def hwfp():
                features = original.freq_e0(original.dwt_hpf(image))
                f1, ll1 = original.freq_down1(features)
                f2, ll2 = original.freq_down2(ll1)
                f3, _ = original.freq_down3(ll2)
                actual = paper.hwfp(features)
                return [compare_tensors(left, right) for left, right in zip((f1, f2, f3), actual, strict=True)]

            check("HWFP_three_recursive_LL_stages", hwfp)

            def msda():
                comparisons = []
                for name, module in original.named_modules():
                    if isinstance(module, original_module.MAC):
                        tensor = torch.randn(1, module.inconv[0].in_channels, 8, 8)
                        other = paper.get_submodule(expected_paper_path(name))
                        comparisons.append({"module": name, **compare_tensors(module(tensor), other(tensor))})
                assert comparisons, "No original MAC modules found."
                return comparisons

            check("MSDA_all_instances", msda)

            def csa():
                comparisons = []
                for name, module in original.named_modules():
                    if isinstance(module, original_module.ResNet):
                        tensor = torch.randn(1, module.conv2.out_channels, 8, 8)
                        old = module.ca(tensor) * tensor
                        old = module.sa(old) * old
                        other = paper.get_submodule(expected_paper_path(name)).csa(tensor)
                        comparisons.append({"module": name, **compare_tensors(old, other)})
                assert comparisons, "No original ResNet attention pairs found."
                return comparisons

            check("CSA_all_channel_then_spatial_attention_pairs", csa)

            def reference_sanity():
                u = torch.tensor([[[2.0, 3.0]]])
                delta = torch.ones_like(u)
                A = torch.zeros(1, 1)
                BC = torch.ones(1, 1, 2)
                actual, final_state = selective_scan_reference(
                    u, delta, A, BC, BC, torch.ones(1), return_last_state=True)
                compare_tensors(actual, torch.tensor([[[4.0, 8.0]]]))
                compare_tensors(final_state, torch.tensor([[[5.0]]]))
                return {"hand_computed_two_step_recurrence": "pass"}

            check("test_only_selective_scan_reference_sanity", reference_sanity)
            original_scan = original_module.selective_scan_fn
            paper_scan = fcsm_module.selective_scan_fn
            original_module.selective_scan_fn = selective_scan_reference
            fcsm_module.selective_scan_fn = selective_scan_reference
            try:
                def fcsm():
                    values = []
                    for old_name, new_name in (("fusion_e3", "fcsm_e3"), ("fusion_d0", "fcsm_d0")):
                        old = getattr(original, old_name)
                        new = getattr(paper, new_name)
                        spatial = torch.randn(1, old.d_model, 4, 4)
                        frequency = torch.randn_like(spatial)
                        values.append({"module": new_name,
                                       **compare_tensors(old(spatial, frequency), new(spatial, frequency))})
                    return values

                check("FCSM_both_instances_with_test_reference", fcsm)
                for deep_supervision in (False, True):
                    def forward(deep_supervision=deep_supervision):
                        old_aux, old_final = original(image, warm_flag=deep_supervision)
                        new_aux, new_final = paper(image, deep_supervision=deep_supervision)
                        assert len(old_aux) == len(new_aux) == (4 if deep_supervision else 0)
                        result = {"input_shape": list(image.shape),
                                  "final": compare_tensors(old_final, new_final),
                                  "auxiliary": [compare_tensors(a, b) for a, b in zip(old_aux, new_aux, strict=True)],
                                  "scan_backend": "test-only CPU reference; not official CUDA"}
                        return result
                    check(f"full_network_deep_supervision_{deep_supervision}", forward)
            finally:
                original_module.selective_scan_fn = original_scan
                fcsm_module.selective_scan_fn = paper_scan
    except Exception as error:
        report["checks"].append({"name": "verification_setup_or_prerequisite", "status": "fail",
                                 "error": f"{type(error).__name__}: {error}",
                                 "traceback": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
    report["passed"] = sum(item["status"] == "pass" for item in report["checks"])
    report["failed"] = sum(item["status"] == "fail" for item in report["checks"])
    report["status"] = "pass" if report["failed"] == 0 else "fail"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: {report['passed']} passed, {report['failed']} failed. Report: {report_path}", flush=True)
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
