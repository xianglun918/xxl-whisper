"""Model catalog and acquisition policy — no cloud-drive dependency.

Primary source is hf-mirror.com (HuggingFace mirror, CN-friendly) serving the
sherpa-onnx team's model exports. Fallback is the GitHub release tarballs.
Each supported model lives in its own directory under the models root.

This module owns *what* to fetch and where it goes (URLs, exact sizes, recorded
SHA256 values, source order, tarball fallback, manual guide). The resumable,
retrying, hash-verified byte transfer lives in :mod:`app.download_io`.
"""

import logging
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.download_io import DownloadError, FileSpec, ProgressFn, verify_sha256
from app.download_io import download as _download

log = logging.getLogger(__name__)

__all__ = [
    "DownloadError",
    "ModelFiles",
    "ProgressFn",
    "ensure_model",
    "ensure_vad_model",
    "manual_download_guide",
]

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

#: Silero VAD artifact for continuous dictation. Two upstream mirrors serve
#: different revisions of the same model: hf-mirror ships the newer 1,807,522-
#: byte export, the GitHub release asset ships the 643,854-byte one. Both
#: segment identically, so either exact size and either known SHA256 is
#: accepted; anything else is a truncated or tampered download. hf-mirror stays
#: primary because it is reachable from mainland China, where GitHub often is not.
_VAD_HF_URL: str = "https://hf-mirror.com/csukuangfj/vad/resolve/main/silero_vad.onnx"
_VAD_HF_SIZE: int = 1_807_522
_VAD_HF_SHA256: str = "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28"
_VAD_GH_URL: str = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)
_VAD_GH_SIZE: int = 643_854
_VAD_GH_SHA256: str = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
_VAD_SIZES: frozenset[int] = frozenset({_VAD_HF_SIZE, _VAD_GH_SIZE})
#: Both upstream revisions are legitimate; either recorded hash is accepted.
_VAD_SHA256: frozenset[str] = frozenset({_VAD_HF_SHA256, _VAD_GH_SHA256})
_VAD_FILENAME: str = "silero_vad.onnx"


@dataclass(frozen=True, slots=True)
class ModelFiles:
    """Resolved local artifacts for one model kind."""

    kind: str
    directory: Path


#: Exact artifact sizes and SHA256 values, recorded from the official upstream
#: files. A truncated or tampered download must never look complete: the size
#: check catches truncation, the hash check catches a poisoned mirror. The
#: known-hash set holds more than one entry only where upstream legitimately
#: serves more than one revision (the Silero VAD).
_MODEL_FILES: dict[str, tuple[tuple[str, str, int, frozenset[str]], ...]] = {
    "sensevoice": (
        (
            f"{_HF_SENSEVOICE}/model.int8.onnx",
            "model.onnx",
            239_233_841,
            frozenset({"c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"}),
        ),
        (
            f"{_HF_SENSEVOICE}/tokens.txt",
            "tokens.txt",
            315_894,
            frozenset({"f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"}),
        ),
    ),
    "funasr_nano": (
        (
            f"{_HF_FUNASR_NANO}/encoder_adaptor.int8.onnx",
            "encoder_adaptor.int8.onnx",
            237_792_748,
            frozenset({"f36dea2e30fbc33b5db1d7a7265cc976c5e5586c77b042d5adb1ad27c72db422"}),
        ),
        (
            f"{_HF_FUNASR_NANO}/embedding.int8.onnx",
            "embedding.int8.onnx",
            155_584_380,
            frozenset({"95e61cd0c9c3b9543339a4cf973c95c116815e745ccc1e0285cbd81f76d18644"}),
        ),
        (
            f"{_HF_FUNASR_NANO}/llm.int8.onnx",
            "llm.int8.onnx",
            600_356_593,
            frozenset({"dfbf9aa3be41bccc257587f151e15c63fbe1b549f2b517f5ccd5bdce3bf4322a"}),
        ),
        (
            f"{_HF_FUNASR_NANO}/Qwen3-0.6B/merges.txt",
            "Qwen3-0.6B/merges.txt",
            1_671_853,
            frozenset({"8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"}),
        ),
        (
            f"{_HF_FUNASR_NANO}/Qwen3-0.6B/tokenizer.json",
            "Qwen3-0.6B/tokenizer.json",
            11_422_654,
            frozenset({"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}),
        ),
        (
            f"{_HF_FUNASR_NANO}/Qwen3-0.6B/vocab.json",
            "Qwen3-0.6B/vocab.json",
            2_776_833,
            frozenset({"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"}),
        ),
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


def ensure_model(
    kind: str, models_root: Path, progress: ProgressFn, *, proxy: str = ""
) -> ModelFiles:
    """Make sure the model's files exist locally; download what is missing."""
    files = _MODEL_FILES[kind]
    model_dir = models_root / kind
    model_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        FileSpec(
            url=url,
            dest=model_dir / dest,
            expected_size=size,
            expected_sha256=sha256,
        )
        for url, dest, size, sha256 in files
    ]
    try:
        for spec in specs:
            if not _is_complete(spec):
                _fetch(spec, progress, proxy=proxy)
    except DownloadError:
        _fetch_tarball_fallback(models_root / kind, specs, progress, proxy=proxy)

    for spec in specs:  # both sources failed for something still missing
        if not _is_complete(spec):
            raise DownloadError(source="all", reason=f"{spec.dest.name} missing")
    return ModelFiles(kind=kind, directory=model_dir)


def ensure_vad_model(
    models_root: Path, progress: ProgressFn, *, proxy: str = ""
) -> Path:
    """Make sure the Silero VAD model exists locally; return its path.

    Continuous dictation downloads this lazily on first enable. hf-mirror is
    the primary source, the GitHub release asset the fallback; the stored file
    must match one of the two known exact sizes and SHA256 values.
    """
    dest = models_root / "vad" / _VAD_FILENAME
    if _is_vad_complete(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: DownloadError | None = None
    for url, size in ((_VAD_HF_URL, _VAD_HF_SIZE), (_VAD_GH_URL, _VAD_GH_SIZE)):
        try:
            _download(
                FileSpec(
                    url=url,
                    dest=dest,
                    expected_size=size,
                    expected_sha256=_VAD_SHA256,
                ),
                progress,
                proxy=proxy,
            )
        except DownloadError as exc:
            last_error = exc
            log.info("VAD source failed, trying next: %s", exc)
        else:
            return dest
    raise last_error if last_error is not None else DownloadError(
        source="all", reason=f"{_VAD_FILENAME} missing"
    )


def manual_download_guide(kind: str, models_root: Path) -> str:
    """Return copy-paste instructions for fetching a model by hand.

    Intranet / proxy-restricted environments may block the in-app downloader;
    this guide lists the exact files and URLs the user must place under the
    model directory so the app proceeds on the next launch.
    """
    model_dir = models_root / kind
    sources = (
        [(_VAD_FILENAME, _VAD_HF_URL), (_VAD_FILENAME, _VAD_GH_URL)]
        if kind == "vad"
        else [(dest, url) for url, dest, _size, _sha256 in _MODEL_FILES[kind]]
    )
    lines = [
        f"模型 {kind} 自动下载失败。",
        "",
        "请手动下载以下文件，按相同目录结构保存到：",
        str(model_dir),
        "",
    ]
    lines.extend(f"{dest}  <-  {url}" for dest, url in sources)
    lines.append("")
    if kind == "vad":
        lines.append("上方两个地址任选其一即可（文件大小 1,807,522 或 643,854 字节）。")
    else:
        lines.append("含子目录的文件（如 Qwen3-0.6B/）需先创建对应目录。")
    lines.append("下载完成后重启 xxl-whisper 即可。")
    return "\n".join(lines)


def _is_complete(spec: FileSpec) -> bool:
    return spec.dest.exists() and spec.dest.stat().st_size == spec.expected_size


def _is_vad_complete(dest: Path) -> bool:
    return dest.exists() and dest.stat().st_size in _VAD_SIZES


def _fetch(spec: FileSpec, progress: ProgressFn, *, proxy: str = "") -> None:
    _download(spec, progress, proxy=proxy)


def _fetch_tarball_fallback(
    model_dir: Path, specs: list[FileSpec], progress: ProgressFn, *, proxy: str = ""
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
            _extract_tarball(tarball_url, model_dir, wanted, progress, proxy=proxy)
        except (OSError, tarfile.TarError) as exc:
            last_error = DownloadError(source=tarball_url, reason=str(exc))
            log.info("tarball source failed, trying next: %s", exc)
        else:
            return
    raise last_error if last_error is not None else DownloadError(
        source="all", reason=f"no tarball fallback for {kind}"
    )


def _member_to_spec(specs: list[FileSpec], kind: str) -> dict[str, FileSpec]:
    """Map every possible tarball member name to its file spec."""
    aliases = _TARBALL_MEMBER_ALIASES.get(kind, {})
    wanted: dict[str, FileSpec] = {spec.dest.name: spec for spec in specs}
    for member_name, dest_name in aliases.items():
        for spec in specs:
            if spec.dest.name == dest_name:
                wanted[member_name] = spec
    return wanted


def _extract_tarball(
    tarball_url: str,
    model_dir: Path,
    wanted: dict[str, FileSpec],
    progress: ProgressFn,
    *,
    proxy: str = "",
) -> None:
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        tar_path = Path(tmp) / "model.tar.bz2"
        _download(
            FileSpec(
                url=tarball_url,
                dest=tar_path,
                expected_size=None,
                expected_sha256=frozenset(),
            ),
            progress,
            proxy=proxy,
        )
        with tarfile.open(tar_path, "r:bz2") as tar:
            for member in tar.getmembers():
                spec = wanted.get(Path(member.name).name)
                if spec is None:
                    continue
                if _is_complete(spec):
                    _discard_part(spec)
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                spec.dest.parent.mkdir(parents=True, exist_ok=True)
                spec.dest.write_bytes(data)
                actual = spec.dest.stat().st_size
                if spec.expected_size is not None and actual != spec.expected_size:
                    spec.dest.unlink(missing_ok=True)
                    raise DownloadError(
                        source=tarball_url,
                        reason=f"extracted {spec.dest.name} has wrong size",
                    )
                digest = verify_sha256(spec.dest, spec.expected_sha256, source=tarball_url)
                log.info(
                    "extracted %s (%d bytes, sha256 %s)",
                    spec.dest.name,
                    actual,
                    digest or "unchecked",
                )
                _discard_part(spec)


def _discard_part(spec: FileSpec) -> None:
    """Remove a stale ``.part`` once the real artifact is in place."""
    spec.dest.with_name(spec.dest.name + ".part").unlink(missing_ok=True)
