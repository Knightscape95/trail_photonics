#!/usr/bin/env python3
"""
colab_state_and_meep.py
──────────────────────────────────────────────────────────────────────────────
• Drive-backed checkpointing that survives Colab disconnects  
• One-time PyMEEP installation in a Drive-hosted Conda environment  
• CLI smoke-tests for both features

Runs unmodified in Colab, local notebooks or CI (fallback stub
directory replaces Drive).  Python ≥ 3.9, MIT licence.
"""
from __future__ import annotations

import argparse
import importlib.metadata as ilm
import logging
import os
import pathlib
import pickle
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

# ─────────────────────────────  LOGGING  ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)8s  %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("colab_state_and_meep")

# ───────────────────────────  GLOBAL STATE  ────────────────────────────────
_drive_mounted = False
_meep_ready = False
_ckpt_cache: Dict[str, pathlib.Path] = {}

_drive_lock = threading.Lock()
_meep_lock = threading.Lock()
_ckpt_lock = threading.Lock()

_BOOTSTRAP_DONE = ".micromamba_bootstrap"
_ENV_READY = ".meep_env_ready"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1 ▸  D R I V E   +   C H E C K P O I N T S                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def mount_drive(drive_path: str = "/content/drive") -> pathlib.Path:
    """Mount Google Drive when inside Colab; otherwise return a stub dir."""
    global _drive_mounted

    with _drive_lock:
        if _drive_mounted:
            root = pathlib.Path(drive_path)
            return root if root.exists() else pathlib.Path.cwd() / "drive_stub"

        try:
            from google.colab import drive as _colab_drive  # type: ignore

            LOG.info("Mounting Google Drive …")
            _colab_drive.mount(drive_path, force_remount=False)
            _drive_mounted = True
            return pathlib.Path(drive_path)
        except ModuleNotFoundError:
            stub = pathlib.Path.cwd() / "drive_stub"
            stub.mkdir(parents=True, exist_ok=True)
            LOG.warning("Colab not detected → using local stub %s", stub)
            _drive_mounted = True
            return stub
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Drive mount failed: {exc}") from exc


def get_ckpt_dir(project: str = "my_project") -> pathlib.Path:
    """Project-specific checkpoint directory (cached per project)."""
    with _ckpt_lock:
        if project in _ckpt_cache and _ckpt_cache[project].exists():
            return _ckpt_cache[project]

        root = mount_drive()
        base = (root / "MyDrive") if (root / "MyDrive").exists() else root
        ckpt = base / f"{project}_ckpts"
        ckpt.mkdir(parents=True, exist_ok=True)
        _ckpt_cache[project] = ckpt
        LOG.info("Checkpoint dir → %s", ckpt)
        return ckpt


# ────────────────  RNG state helpers  ────────────────
def _collect_rng() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except ModuleNotFoundError:
        pass
    try:
        import torch  # type: ignore

        state["torch_cpu"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.random.get_rng_state_all()
    except ModuleNotFoundError:
        pass
    return state


def _restore_rng(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np  # type: ignore

        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        import torch  # type: ignore

        torch.random.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(state["torch_cuda"])


# ────────────────  Checkpoint I/O  ────────────────
def save_state(step: int, project: str = "my_project", **objects: Any) -> None:
    """Atomic pickle of objects + RNG into project checkpoint folder."""
    ckpt_dir = get_ckpt_dir(project)
    fname = ckpt_dir / f"ckpt_step_{step:06d}.pkl"
    tmp = fname.with_suffix(".tmp")

    payload = {"__version__": 1, "step": step, "objects": objects, "rng": _collect_rng()}

    try:
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(fname)
        LOG.debug("Checkpoint saved → %s", fname)
    except Exception as exc:  # pylint: disable=broad-except
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint save failed: {exc}") from exc


def load_latest_state(project: str = "my_project") -> Tuple[Optional[Dict[str, Any]], int]:
    """Return (objects, step) or (None, –1). Corrupt files renamed *.corrupt."""
    ckpt_dir = get_ckpt_dir(project)
    files = sorted(ckpt_dir.glob("ckpt_step_*.pkl"))
    if not files:
        return None, -1

    latest = files[-1]
    try:
        with latest.open("rb") as fh:
            data = pickle.load(fh)
        if data.get("__version__") != 1:
            LOG.warning("Checkpoint version mismatch—ignored.")
            return None, -1
        _restore_rng(data["rng"])
        LOG.info("Resumed from %s (step %d)", latest.name, data["step"])
        return data["objects"], int(data["step"])
    except Exception as exc:  # pylint: disable=broad-except
        LOG.error("Could not load %s (%s) → marking corrupt.", latest, exc)
        latest.rename(latest.with_suffix(".corrupt"))
        return None, -1


# ────────────────  Demo loop  ────────────────
def train_demo(project: str = "demo_project", total: int = 300, every: int = 25) -> None:
    objs, start = load_latest_state(project)
    step = start + 1
    LOG.info("Demo starts at step %d / %d", step, total)

    try:
        for step in range(step, total):
            loss, acc = random.random(), 0.8 + random.random() * 0.2
            time.sleep(1e-4)

            if step % every == 0 and step:
                save_state(step, project, metrics=dict(loss=loss, acc=acc), extras=objs)

            if step % 100 == 0:
                LOG.info("step=%4d  loss=%.3f  acc=%.3f", step, loss, acc)
        LOG.info("Demo finished.")
    except KeyboardInterrupt:
        LOG.warning("Interrupted → checkpointing.")
        save_state(step, project, interrupted=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2 ▸  O N E-T I M E   P y M E E P   I N S T A L L                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def ensure_meep(env: str = "meep_env", version: str = "1.27.0") -> None:
    """Import PyMEEP, installing/activating env if needed (thread-safe)."""
    global _meep_ready
    with _meep_lock:
        if _meep_ready:
            return
        try:
            present = ilm.version("meep")
            if present != version:
                LOG.warning("PyMEEP %s already in env (requested %s).", present, version)
            _meep_ready = True
            return
        except ilm.PackageNotFoundError:
            pass  # install path below

        drive = mount_drive()
        env_root = drive / "MyDrive" / "conda_envs" / env
        env_flag = env_root / _ENV_READY

        _ensure_micromamba(drive)
        if not env_flag.exists() or not shutil.which("micromamba"):
            _install_meep_env(env_root, version)
            env_flag.touch()

        _activate_conda_env(env_root)

        try:
            import meep as mp  # type: ignore
            LOG.info("PyMEEP ready (v%s).", mp.__version__)
            _meep_ready = True
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("PyMEEP import failed post-install.") from exc


# ────────────────  micromamba bootstrap  ────────────────
def _ensure_micromamba(drive_root: pathlib.Path) -> None:
    sent = drive_root / _BOOTSTRAP_DONE
    if shutil.which("micromamba"):
        return
    if sent.exists():
        LOG.info("micromamba bootstrap recorded but binary missing → re-bootstrapping.")

    LOG.info("Installing micromamba via condacolab (kernel may restart).")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "condacolab"])
    import condacolab  # type: ignore

    sent.touch()
    if "google.colab" in sys.modules:
        condacolab.install()            # triggers restart
        sys.exit(0)
    condacolab.install()                # no restart outside Colab


# ────────────────  Conda env creation  ────────────────
def _install_meep_env(env_path: pathlib.Path, meep_ver: str) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("micromamba"):
        raise RuntimeError("micromamba missing after bootstrap.")

    LOG.info("Creating PyMEEP env at %s …", env_path)
    subprocess.check_call(
        [
            "micromamba",
            "create",
            "-y",
            "-p",
            str(env_path),
            "-c",
            "conda-forge",
            "python=3.10",
            f"pymeep={meep_ver}",
            "mpb",
            "numpy",
            "scipy",
            "h5py",
        ]
    )


# ────────────────  Env activation  ────────────────
def _activate_conda_env(env: pathlib.Path) -> None:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    sp = env / "lib" / f"python{pyver}" / "site-packages"
    lib = env / "lib"
    if not sp.exists():
        raise RuntimeError("site-packages not found in env.")

    if str(sp) not in sys.path:
        sys.path.insert(0, str(sp))
    import site  # type: ignore

    site.addsitedir(str(sp))

    sep = os.pathsep
    os.environ["PATH"] = f"{env / 'bin'}{sep}{os.environ.get('PATH', '')}"
    if sys.platform == "darwin":
        os.environ["DYLD_LIBRARY_PATH"] = f"{lib}{sep}{os.environ.get('DYLD_LIBRARY_PATH', '')}"
    elif os.name == "nt":
        os.environ["PATH"] = f"{lib}{sep}{os.environ['PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{lib}{sep}{os.environ.get('LD_LIBRARY_PATH', '')}"


# ────────────────  Smoke test  ────────────────
def run_meep_test() -> bool:
    try:
        import meep as mp  # type: ignore

        LOG.info("Running PyMEEP smoke test …")
        sim = mp.Simulation(
            cell_size=mp.Vector3(4, 2),
            boundary_layers=[mp.PML(0.5)],
            resolution=10,
            sources=[mp.Source(mp.ContinuousSource(0.15), mp.Ez, center=mp.Vector3(-1, 0))],
        )
        sim.run(until=10)
        LOG.info("Smoke test OK.")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        LOG.error("Smoke test failed: %s", exc)
        return False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3 ▸  C L I                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Drive-backed checkpointing + one-time PyMEEP installer",
    )
    p.add_argument("--project", default="demo_project", help="checkpoint namespace")
    p.add_argument("--env-name", default="meep_env", help="Conda env name")
    p.add_argument("--meep-version", default="1.27.0", help="desired PyMEEP version")
    p.add_argument("--demo", action="store_true", help="run checkpoint + meep tests")
    p.add_argument("--checkpoint", action="store_true", help="checkpoint demo only")
    p.add_argument("--meep", action="store_true", help="PyMEEP smoke test only")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main() -> None:  # pragma: no cover
    args = _cli().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not (args.demo or args.checkpoint or args.meep):
        LOG.error("Choose at least one action: --demo / --checkpoint / --meep")
        sys.exit(2)

    if args.demo or args.checkpoint:
        train_demo(args.project)

    if args.demo or args.meep:
        ensure_meep(args.env_name, args.meep_version)
        run_meep_test()


if __name__ == "__main__":
    main()
