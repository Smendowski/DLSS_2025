#!/usr/bin/env python3
"""
check_env.py — verify installed packages against requirements, check Python version,
and test GPU backends.

Usage:
  python check_env.py [path/to/requirements.txt]

Color code:
  green  = version satisfies requirement (or matches exact pin)
  red    = version is too old / missing / does not satisfy requirement
  orange = version is newer than an exact pin (==) or above an upper bound (<=)
"""

import argparse
import sys
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

# --- Colors (ANSI) ---
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
ORANGE = "\033[33m"  # amber/yellow
CYAN = "\033[36m"
BOLD = "\033[1m"

# --- Packaging & metadata imports (with graceful fallback) ---
try:
    from importlib import metadata as importlib_metadata  # Py3.8+
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version, InvalidVersion
    from packaging.utils import canonicalize_name
except Exception:
    sys.stderr.write(
        f"{RED}This script requires the 'packaging' library."
        f"\nInstall it with: pip install packaging{RESET}\n"
    )
    sys.exit(2)


@dataclass
class CheckResult:
    name: str
    required: str
    installed: Optional[str]
    status: str     # "ok", "older", "newer", "missing", "unsatisfied"
    message: str


def colorize(s: str, status: str) -> str:
    if status == "ok":
        return f"{GREEN}{s}{RESET}"
    if status in ("older", "missing", "unsatisfied"):
        return f"{RED}{s}{RESET}"
    if status == "newer":
        return f"{ORANGE}{s}{RESET}"
    return s


def load_requirements(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Skip VCS/editable/constraints and options for simplicity.
            if line.startswith(("-e ", "--", "-c ", "-f ", "-i ", "--extra-index-url")):
                continue
            # Support nested requirements via "-r file.txt"
            if line.startswith("-r "):
                nested = line.split(None, 1)[1]
                try:
                    lines.extend(load_requirements(nested))
                except FileNotFoundError:
                    print(colorize(f"Nested requirements file not found: {nested}", "unsatisfied"))
                continue
            lines.append(line)
    return lines


def find_installed_version(dist_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        pass
    want = canonicalize_name(dist_name)
    for d in importlib_metadata.distributions():
        try:
            if canonicalize_name(d.metadata["Name"]) == want:
                return d.version
        except Exception:
            continue
    return None


def compare_versions(req: Requirement, installed: Optional[str]) -> Tuple[str, str]:
    name = req.name
    spec: SpecifierSet = req.specifier

    if installed is None:
        return "missing", "not installed"

    if not spec:
        return "ok", "no constraint"

    try:
        installed_v = Version(installed)
    except InvalidVersion:
        return ("ok" if installed in str(spec) else "unsatisfied",
                f"installed={installed}, spec={spec}")

    exact_pins = [s for s in spec if s.operator == "=="]
    if exact_pins:
        targets = [Version(s.version) for s in exact_pins if _safe_version(s.version)]
        if any(installed_v == t for t in targets):
            return "ok", f"matches {spec}"
        if all(installed_v > t for t in targets):
            return "newer", f"{installed} > pinned {spec}"
        else:
            return "older", f"{installed} < pinned {spec}"

    ge_list = [Version(s.version) for s in spec if s.operator in (">=", ">") and _safe_version(s.version)]
    le_list = [Version(s.version) for s in spec if s.operator in ("<=", "<") and _safe_version(s.version)]

    satisfies = installed_v in spec

    if ge_list and not le_list:
        lower = max(ge_list)
        if installed_v < lower:
            return "older", f"{installed} < minimum {lower}"
        return "ok", f"satisfies {spec}"

    if le_list and not ge_list:
        upper = min(le_list)
        if installed_v > upper:
            return "newer", f"{installed} > maximum {upper}"
        return "ok", f"satisfies {spec}"

    if ge_list and le_list:
        lower = max(ge_list)
        upper = min(le_list)
        if not satisfies:
            if installed_v < lower:
                return "older", f"{installed} < minimum {lower}"
            if installed_v > upper:
                return "newer", f"{installed} > maximum {upper}"
            return "unsatisfied", f"does not satisfy {spec}"
        return "ok", f"satisfies {spec}"

    if satisfies:
        return "ok", f"satisfies {spec}"
    else:
        approx = [s for s in spec if s.operator == "~="]
        if approx:
            base = approx[0].version
            return "unsatisfied", f"{installed} does not satisfy ~= {base}"
        return "unsatisfied", f"does not satisfy {spec}"


def _safe_version(v: str) -> bool:
    try:
        Version(v)
        return True
    except InvalidVersion:
        return False


def check_requirements(req_lines: List[str]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for line in req_lines:
        try:
            req = Requirement(line)
        except Exception:
            results.append(CheckResult(line, line, None, "unsatisfied", "could not parse requirement line"))
            continue

        installed = find_installed_version(req.name)
        status, msg = compare_versions(req, installed)
        results.append(
            CheckResult(
                name=req.name,
                required=str(req.specifier) if req.specifier else "(any)",
                installed=installed,
                status=status,
                message=msg,
            )
        )
    return results


def print_results(results: List[CheckResult]) -> None:
    name_w = max([len("Package")] + [len(r.name) for r in results])
    req_w = max([len("Required")] + [len(r.required) for r in results])
    inst_w = max([len("Installed")] + [len(r.installed or "—") for r in results])

    header = f"{BOLD}{'Package'.ljust(name_w)}  {'Required'.ljust(req_w)}  {'Installed'.ljust(inst_w)}  Status  Details{RESET}"
    print(header)
    print("-" * len(header))

    counts = {"ok": 0, "older": 0, "newer": 0, "missing": 0, "unsatisfied": 0}
    for r in results:
        counts[r.status] += 1
        status_colored = {
            "ok": colorize("OK", "ok"),
            "older": colorize("OLDER", "older"),
            "newer": colorize("NEWER", "newer"),
            "missing": colorize("MISSING", "missing"),
            "unsatisfied": colorize("UNSAT", "unsatisfied"),
        }[r.status]
        print(
            f"{r.name.ljust(name_w)}  {r.required.ljust(req_w)}  {(r.installed or '—').ljust(inst_w)}  "
            f"{status_colored}   {r.message}"
        )

    print()
    print(f"{BOLD}Summary:{RESET} "
          f"{colorize(str(counts['ok'])+' ok', 'ok')}, "
          f"{colorize(str(counts['older'])+' older', 'older')}, "
          f"{colorize(str(counts['newer'])+' newer', 'newer')}, "
          f"{colorize(str(counts['missing'])+' missing', 'missing')}, "
          f"{colorize(str(counts['unsatisfied'])+' unsatisfied', 'unsatisfied')}")


# ---------------- Python version check ----------------

def check_python_version():
    print(f"{BOLD}{CYAN}Python version check{RESET}")
    vi = sys.version_info
    version_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    if vi.major == 3 and vi.minor == 12:
        print(colorize(f"Python {version_str} (3.12.x)", "ok"))
    else:
        print(colorize(f"Python {version_str} (expected 3.12.x)", "unsatisfied"))
    print()


# ---------------- GPU CHECKS ----------------

def check_pytorch():
    print(f"\n{BOLD}{CYAN}PyTorch GPU check{RESET}")
    try:
        import torch  # type: ignore
    except Exception:
        print(colorize("PyTorch not installed.", "missing"))
        return

    print(f"torch.__version__ = {torch.__version__}")
    try:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(colorize(f"CUDA available: {n} device(s)", "ok"))
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                cap = ".".join(map(str, torch.cuda.get_device_capability(i)))
                print(f"  - cuda:{i} -> {name} (cc {cap})")
            x = torch.randn((128, 128), device="cuda:0")
            y = x @ x
            _ = y.mean().item()
            print(colorize("CUDA tensor op: OK", "ok"))
        else:
            print(colorize("CUDA not available", "unsatisfied"))
            if getattr(torch.version, "cuda", None):
                print(f"  (Built with CUDA {torch.version.cuda})")
    except Exception as e:
        print(colorize(f"CUDA check error: {e}", "unsatisfied"))

    try:
        mps_built = getattr(torch.backends.mps, "is_built", lambda: False)()
        mps_ok = getattr(torch.backends.mps, "is_available", lambda: False)()
        if mps_built:
            print(f"MPS built: {mps_built}")
        if mps_ok:
            print(colorize("Metal/MPS available", "ok"))
            try:
                x = torch.randn((128, 128), device="mps")
                y = x @ x
                _ = y.mean().item()
                print(colorize("MPS tensor op: OK", "ok"))
            except Exception as e:
                print(colorize(f"MPS tensor op failed: {e}", "unsatisfied"))
        else:
            print("Metal/MPS not available")
    except Exception as e:
        print(colorize(f"MPS check error: {e}", "unsatisfied"))


def check_tensorflow():
    print(f"\n{BOLD}{CYAN}TensorFlow GPU check{RESET}")
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        print(colorize("TensorFlow not installed.", "missing"))
        return

    print(f"tf.__version__ = {tf.__version__}")
    try:
        phys = tf.config.list_physical_devices("GPU")
        logi = tf.config.list_logical_devices("GPU")
        if phys:
            print(colorize(f"GPUs (physical): {len(phys)}", "ok"))
            for d in phys:
                print(f"  - {d}")
        else:
            print(colorize("No physical GPU devices visible", "unsatisfied"))
        if logi:
            print(f"GPUs (logical): {len(logi)}")
            for d in logi:
                print(f"  - {d}")
        if logi:
            try:
                with tf.device("/GPU:0"):
                    a = tf.random.normal((128, 128))
                    b = tf.random.normal((128, 128))
                    c = tf.matmul(a, b)
                    _ = tf.reduce_mean(c).numpy()
                print(colorize("Tensor op on /GPU:0: OK", "ok"))
            except Exception as e:
                print(colorize(f"Tensor op on /GPU:0 failed: {e}", "unsatisfied"))
        try:
            built_cuda = tf.test.is_built_with_cuda()
            print(f"Built with CUDA: {built_cuda}")
        except Exception:
            pass
    except Exception as e:
        print(colorize(f"TensorFlow GPU check error: {e}", "unsatisfied"))


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Check environment vs requirements and GPU backends.")
    ap.add_argument("requirements", nargs="?", default="requirements.txt",
                    help="Path to requirements file (default: requirements.txt)")
    args = ap.parse_args()

    check_python_version()   # strict 3.12.x check

    try:
        lines = load_requirements(args.requirements)
    except FileNotFoundError:
        print(colorize(f"Requirements file not found: {args.requirements}", "unsatisfied"))
        sys.exit(1)

    results = check_requirements(lines)
    print_results(results)

    check_pytorch()
    check_tensorflow()


if __name__ == "__main__":
    main()
