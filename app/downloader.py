"""Model acquisition: direct downloads — no cloud-drive dependency.

Primary source is hf-mirror.com (HuggingFace mirror, CN-friendly) serving the
sherpa-onnx team's SenseVoice export, which is the build paired with the
sherpa-onnx wheel's bundled onnxruntime. Fallback is the GitHub release
tarball. Files land under %LOCALAPPDATA%/xxl-whisper/models/sensevoice/.
"""

import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_HF_BASE: str = (
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    "/resolve/main"
)
_GH_TARBALL: str = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    "/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
)
_MODEL_NAME: str = "model.onnx"
_TOKENS_NAME: str = "tokens.txt"

#: Exact artifact sizes — a truncated download must never look complete.
MODEL_SIZE: int = 239_233_841  # model.int8.onnx from the sherpa-onnx export
TOKENS_SIZE: int = 315_894

ProgressFn = Callable[[str, int, int], None]  # (filename, downloaded_bytes, total_bytes)


class DownloadError(Exception):
    """Raised when model files cannot be fetched from any source."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ModelFiles:
    model: Path
    tokens: Path


@dataclass(frozen=True, slots=True)
class _Spec:
    url: str
    dest: Path
    expected_size: int


def ensure_model(model_root: Path, progress: ProgressFn) -> ModelFiles:
    """Make sure model + tokens exist locally; download what is missing."""
    model_root.mkdir(parents=True, exist_ok=True)
    specs = [
        _Spec(
            url=f"{_HF_BASE}/model.int8.onnx",
            dest=model_root / _MODEL_NAME,
            expected_size=MODEL_SIZE,
        ),
        _Spec(
            url=f"{_HF_BASE}/tokens.txt",
            dest=model_root / _TOKENS_NAME,
            expected_size=TOKENS_SIZE,
        ),
    ]
    try:
        for spec in specs:
            if not _is_complete(spec):
                _fetch(spec, progress)
    except DownloadError:
        _fetch_tarball_fallback(model_root, specs, progress)

    for spec in specs:  # both sources failed for something still missing
        if not _is_complete(spec):
            raise DownloadError(source="all", reason=f"{spec.dest.name} missing")
    return ModelFiles(model=specs[0].dest, tokens=specs[1].dest)


def _is_complete(spec: _Spec) -> bool:
    return spec.dest.exists() and spec.dest.stat().st_size == spec.expected_size


def _fetch(spec: _Spec, progress: ProgressFn) -> None:
    _download(
        url=spec.url,
        dest=spec.dest,
        display_name=spec.dest.name,
        progress=progress,
        expected_size=spec.expected_size,
    )


def _download(
    url: str,
    dest: Path,
    display_name: str,
    progress: ProgressFn,
    expected_size: int | None,
) -> None:
    """Stream one URL to dest via a .part file; verify size when known."""
    part = dest.with_name(dest.name + ".part")
    try:
        with (
            # URLs are module-level https constants, not user input.
            urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310
            part.open("wb") as out,
        ):
            header_size = response.headers.get("Content-Length")
            total = int(header_size) if header_size else (expected_size or 0)
            downloaded = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                downloaded += len(chunk)
                progress(display_name, downloaded, total)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise DownloadError(source=url, reason=str(exc)) from exc
    if expected_size is not None and part.stat().st_size != expected_size:
        actual = part.stat().st_size
        part.unlink(missing_ok=True)
        raise DownloadError(source=url, reason=f"size mismatch: {actual}")
    part.replace(dest)


def _fetch_tarball_fallback(
    model_root: Path, specs: list[_Spec], progress: ProgressFn
) -> None:
    """Extract the two files from the GitHub release tarball."""
    wanted = {"model.int8.onnx": specs[0], "tokens.txt": specs[1]}
    try:
        with tempfile.TemporaryDirectory(dir=model_root) as tmp:
            tar_path = Path(tmp) / "model.tar.bz2"
            _download(
                url=_GH_TARBALL,
                dest=tar_path,
                display_name="model.tar.bz2",
                progress=progress,
                expected_size=None,
            )
            with tarfile.open(tar_path, "r:bz2") as tar:
                for member in tar.getmembers():
                    spec = wanted.get(Path(member.name).name)
                    if spec is None or _is_complete(spec):
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    data = extracted.read()
                    spec.dest.write_bytes(data)
                    if spec.dest.stat().st_size != spec.expected_size:
                        spec.dest.unlink(missing_ok=True)
                        raise DownloadError(
                            source=_GH_TARBALL,
                            reason=f"extracted {spec.dest.name} has wrong size",
                        )
    except (OSError, tarfile.TarError) as exc:
        raise DownloadError(source=_GH_TARBALL, reason=str(exc)) from exc
