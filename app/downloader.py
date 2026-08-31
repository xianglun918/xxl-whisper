"""Model acquisition: direct downloads — no cloud-drive dependency.

Primary source is hf-mirror.com (HuggingFace mirror, CN-friendly) serving the
sherpa-onnx team's model exports. Fallback is the GitHub release tarballs.
Each supported model lives in its own directory under the models root.
"""

import logging
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_HF_SENSEVOICE: str = (
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    "/resolve/main"
)
_HF_FUNASR_NANO: str = (
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-funasr-nano-int8-2025-12-30"
    "/resolve/main"
)
_GH_TARBALL_SENSEVOICE: str = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    "/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
)
_GH_TARBALL_FUNASR_NANO: str = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    "/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2"
)
#: Our own backup of the default model, independent of hf-mirror and k2-fsa.
_GH_BACKUP_SENSEVOICE: str = (
    "https://github.com/xianglun918/xxl-whisper/releases/download/models"
    "/sensevoice-backup.tar.bz2"
)

ProgressFn = Callable[[str, int, int], None]  # (filename, downloaded_bytes, total_bytes)


class DownloadError(Exception):
    """Raised when model files cannot be fetched from any source."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ModelFiles:
    """Resolved local artifacts for one model kind."""

    kind: str
    directory: Path


@dataclass(frozen=True, slots=True)
class _FileSpec:
    url: str
    dest: Path
    expected_size: int


#: Exact artifact sizes — a truncated download must never look complete.
_MODEL_FILES: dict[str, tuple[tuple[str, str, int], ...]] = {
    "sensevoice": (
        (f"{_HF_SENSEVOICE}/model.int8.onnx", "model.onnx", 239_233_841),
        (f"{_HF_SENSEVOICE}/tokens.txt", "tokens.txt", 315_894),
    ),
    "funasr_nano": (
        (f"{_HF_FUNASR_NANO}/encoder_adaptor.int8.onnx", "encoder_adaptor.int8.onnx", 237_792_748),
        (f"{_HF_FUNASR_NANO}/embedding.int8.onnx", "embedding.int8.onnx", 155_584_380),
        (f"{_HF_FUNASR_NANO}/llm.int8.onnx", "llm.int8.onnx", 600_356_593),
        (f"{_HF_FUNASR_NANO}/Qwen3-0.6B/merges.txt", "Qwen3-0.6B/merges.txt", 1_671_853),
        (f"{_HF_FUNASR_NANO}/Qwen3-0.6B/tokenizer.json", "Qwen3-0.6B/tokenizer.json", 11_422_654),
        (f"{_HF_FUNASR_NANO}/Qwen3-0.6B/vocab.json", "Qwen3-0.6B/vocab.json", 2_776_833),
    ),
}

_MODEL_TARBALLS: dict[str, tuple[str, ...]] = {
    "sensevoice": (_GH_TARBALL_SENSEVOICE, _GH_BACKUP_SENSEVOICE),
    "funasr_nano": (_GH_TARBALL_FUNASR_NANO,),
}

#: Tarball member name -> on-disk filename, where upstream tarballs use a
#: different name than we store (k2-fsa ships model.int8.onnx; we save model.onnx).
_TARBALL_MEMBER_ALIASES: dict[str, dict[str, str]] = {
    "sensevoice": {"model.int8.onnx": "model.onnx"},
    "funasr_nano": {},
}


def ensure_model(kind: str, models_root: Path, progress: ProgressFn) -> ModelFiles:
    """Make sure the model's files exist locally; download what is missing."""
    files = _MODEL_FILES[kind]
    model_dir = models_root / kind
    model_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        _FileSpec(url=url, dest=model_dir / dest, expected_size=size)
        for url, dest, size in files
    ]
    try:
        for spec in specs:
            if not _is_complete(spec):
                _fetch(spec, progress)
    except DownloadError:
        _fetch_tarball_fallback(models_root / kind, specs, progress)

    for spec in specs:  # both sources failed for something still missing
        if not _is_complete(spec):
            raise DownloadError(source="all", reason=f"{spec.dest.name} missing")
    return ModelFiles(kind=kind, directory=model_dir)


def manual_download_guide(kind: str, models_root: Path) -> str:
    """Return copy-paste instructions for fetching a model by hand.

    Intranet / proxy-restricted environments may block the in-app downloader;
    this guide lists the exact files and URLs the user must place under the
    model directory so the app proceeds on the next launch.
    """
    model_dir = models_root / kind
    lines = [
        f"模型 {kind} 自动下载失败。",
        "",
        "请手动下载以下文件，按相同目录结构保存到：",
        str(model_dir),
        "",
    ]
    for url, dest, _size in _MODEL_FILES[kind]:
        lines.append(f"{dest}  <-  {url}")
    lines.append("")
    lines.append("含子目录的文件（如 Qwen3-0.6B/）需先创建对应目录。")
    lines.append("下载完成后重启 xxl-whisper 即可。")
    return "\n".join(lines)


def _is_complete(spec: _FileSpec) -> bool:
    return spec.dest.exists() and spec.dest.stat().st_size == spec.expected_size


def _fetch(spec: _FileSpec, progress: ProgressFn) -> None:
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
    dest.parent.mkdir(parents=True, exist_ok=True)
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
    model_dir: Path, specs: list[_FileSpec], progress: ProgressFn
) -> None:
    """Extract model files from a release tarball, trying each source in order."""
    kind = model_dir.name
    tarball_urls = _MODEL_TARBALLS.get(kind)
    if not tarball_urls:
        raise DownloadError(source="all", reason=f"no tarball fallback for {kind}")
    wanted = _member_to_spec(specs, kind)
    last_error: DownloadError | None = None
    for tarball_url in tarball_urls:
        try:
            _extract_tarball(tarball_url, model_dir, wanted, progress)
        except (OSError, tarfile.TarError) as exc:
            last_error = DownloadError(source=tarball_url, reason=str(exc))
            log.info("tarball source failed, trying next: %s", exc)
        else:
            return
    raise last_error if last_error is not None else DownloadError(
        source="all", reason=f"no tarball fallback for {kind}"
    )


def _member_to_spec(specs: list[_FileSpec], kind: str) -> dict[str, _FileSpec]:
    """Map every possible tarball member name to its file spec."""
    aliases = _TARBALL_MEMBER_ALIASES.get(kind, {})
    wanted: dict[str, _FileSpec] = {spec.dest.name: spec for spec in specs}
    for member_name, dest_name in aliases.items():
        for spec in specs:
            if spec.dest.name == dest_name:
                wanted[member_name] = spec
    return wanted


def _extract_tarball(
    tarball_url: str,
    model_dir: Path,
    wanted: dict[str, _FileSpec],
    progress: ProgressFn,
) -> None:
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        tar_path = Path(tmp) / "model.tar.bz2"
        _download(
            url=tarball_url,
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
                spec.dest.parent.mkdir(parents=True, exist_ok=True)
                spec.dest.write_bytes(data)
                if spec.dest.stat().st_size != spec.expected_size:
                    spec.dest.unlink(missing_ok=True)
                    raise DownloadError(
                        source=tarball_url,
                        reason=f"extracted {spec.dest.name} has wrong size",
                    )
