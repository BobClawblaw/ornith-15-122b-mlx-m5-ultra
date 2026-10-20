"""Settings: recipe.toml's defaults, local.toml's overrides for this Mac, then environment variables in upper case."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    source_repo: str = ""
    source_revision: str = ""
    served_name: str = ""
    model_dir: str = ""
    bits: int = 4
    group_size: int = 64
    python: str = ".venv/bin/python"
    host: str = "127.0.0.1"
    port: int = 8891
    cache: str = "~/.cache/ornith-15-122b"
    max_num_seqs: int = 4
    max_tokens: int = 32768
    max_request_tokens: int = 262144
    prefix_cache: bool = True
    mtp: bool = False
    mtp_draft_tokens: int = 3
    kv_cache_bits: int = 0
    reasoning_parser: str = "qwen3"
    extra_args: str = ""
    memory_margin_gib: float = 8

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @property
    def model(self) -> Path:
        return self.path(self.model_dir)

    @property
    def cache_dir(self) -> Path:
        return self.path(self.cache)

    @property
    def interpreter(self) -> Path:
        p = self.path(self.python)
        return p if p.is_absolute() else ROOT / p

    @property
    def vllm(self) -> Path:
        return self.interpreter.parent / "vllm-mlx"

    @property
    def drafter(self) -> Path:
        """The MTP layer `convert --mtp` extracts beside the checkpoint (vllm-mlx's multimodal drafter)."""
        return self.model.parent / (self.model.name + "-mtp")

    @property
    def source_snapshot(self) -> Path:
        return self.path(f"~/.cache/huggingface/hub/models--{self.source_repo.replace('/', '--')}/snapshots/{self.source_revision}")

    @property
    def pid_file(self) -> Path:
        return self.cache_dir / "server.pid"

    @property
    def log_file(self) -> Path:
        return self.cache_dir / "server.log"

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"

    def serve_args(self) -> list[str]:
        a = [str(self.vllm), "serve", str(self.model), "--served-model-name", self.served_name, "--mllm",
             "--host", self.host, "--port", str(self.port), "--continuous-batching", "--max-num-seqs", str(self.max_num_seqs),
             "--max-tokens", str(self.max_tokens), "--max-request-tokens", str(self.max_request_tokens),
             "--enable-auto-tool-choice", "--tool-call-parser", "qwen3.5"]
        a.append("--enable-prefix-cache" if self.prefix_cache else "--disable-prefix-cache")
        if self.mtp:
            a += ["--mllm-draft-model", str(self.drafter), "--mllm-draft-kind", "mtp", "--mllm-draft-block-size",
                  str(self.mtp_draft_tokens), "--default-mllm-draft"]
        if self.kv_cache_bits in (4, 8):
            a += ["--kv-cache-quantization", "--kv-cache-quantization-bits", str(self.kv_cache_bits)]
        if self.reasoning_parser:
            a += ["--reasoning-parser", self.reasoning_parser]
        a += self.extra_args.split()
        return a

    def convert_args(self, bits: int) -> tuple[list[str], Path]:
        out = self.path(f"~/models/Ornith-1.5-122B-MLX-{bits}bit")
        return [str(self.interpreter), "-m", "mlx_vlm.convert", "--hf-path", str(self.source_snapshot), "--mlx-path", str(out),
                "-q", "--q-bits", str(bits), "--q-group-size", str(self.group_size), "--q-mode", "affine", "--dtype", "bfloat16",
                "--mtp"], out


def _coerce(kind: type, raw):
    if kind is bool:
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return kind(raw)


def load(local_file: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    merged: dict = {}
    for f in (ROOT / "recipe.toml", local_file or ROOT / "local.toml"):
        if f.exists():
            for section in tomllib.loads(f.read_text()).values():
                merged.update(section)
    s = Settings()
    for f in fields(Settings):
        if f.name in merged:
            setattr(s, f.name, _coerce(f.type if isinstance(f.type, type) else type(getattr(s, f.name)), merged[f.name]))
        if f.name.upper() in env:
            setattr(s, f.name, _coerce(type(getattr(s, f.name)), env[f.name.upper()]))
    return s


def validate(s: Settings) -> list[str]:
    problems = []
    if not s.interpreter.exists():
        problems.append(f"no interpreter at {s.interpreter}: python3 -m mac venv")
    if not (s.model / "config.json").exists():
        problems.append(f"no converted checkpoint at {s.model}: python3 -m mac convert")
    if s.mtp and not (s.drafter / "config.json").exists():
        problems.append(f"mtp = true but there is no drafter at {s.drafter} (convert writes it; or set mtp = false)")
    if not s.vllm.exists():
        problems.append(f"no vllm-mlx at {s.vllm}: python3 -m mac venv")
    return problems
