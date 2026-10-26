"""Settings: recipe.toml's defaults, local.toml's overrides for this Mac, then environment variables in upper case."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JPEG_VERSION = "3.1.4.1"  # the libjpeg-turbo release the engine's JPEG decoder pins (Pillow's, bit for bit)


@dataclass
class Settings:
    source_repo: str = ""
    source_revision: str = ""
    served_name: str = ""
    model_dir: str = ""
    drafter_repo: str = ""
    drafter_revision: str = ""
    drafter_dir: str = ""
    drafter_bits: int = 4
    tensorfold_src: str = ""
    tensorfold_sha: str = ""
    zig: str = "zig"
    zig_cpu: str = "native"
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 8891
    tf_cache: str = "~/.cache/ornith-15-122b"
    context: int = 262144
    parallel: int = 4
    drafts: bool = True
    thinking: bool = False
    max_tokens: int = 50176
    thinking_budget: int = 49152
    temperature: str = ""
    prompt_cache_gib: float = 32
    prompt_cache_over_cap: bool = False
    vision: bool = True
    vision_image_tokens: int = 4096
    vision_max_images: int = 4
    vision_urls: bool = False
    keep_warm: int = 86400
    api_key: str = ""
    extra_args: str = ""
    memory_margin_gib: float = 8

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @staticmethod
    def snapshot(repo: str, revision: str) -> Path:
        return Settings.path(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/{revision}")

    @property
    def model(self) -> Path:
        return self.path(self.model_dir)

    @property
    def drafter(self) -> Path:
        """The DFlash2 drafter: a directory of its own, or the pinned release in the Hugging Face cache."""
        return self.path(self.drafter_dir) if self.drafter_dir else self.snapshot(self.drafter_repo, self.drafter_revision)

    @property
    def cache(self) -> Path:
        return self.path(self.tf_cache)

    @property
    def worktree(self) -> Path:
        return self.cache / "engine" / self.tensorfold_sha[:12]

    @property
    def built_binary(self) -> Path:
        return self.worktree / "zig-out" / "native" / "bin" / "tensorfold-native"

    @property
    def built_grammar(self) -> Path:
        """libtfgrammar (xgrammar), which the server loads beside its binary (../lib) at the first structured request."""
        return self.worktree / "zig-out" / "native" / "lib" / "libtfgrammar.dylib"

    @property
    def run_binary(self) -> Path:
        return self.path(self.binary) if self.binary else self.built_binary

    @property
    def jpeg_lib(self) -> Path:
        return self.cache / f"libjpeg-turbo-{JPEG_VERSION}" / "lib" / "libturbojpeg.0.dylib"

    @property
    def pid_file(self) -> Path:
        return self.cache / "server.pid"

    @property
    def log_file(self) -> Path:
        return self.cache / "server.log"

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"

    def serve_args(self) -> list[str]:
        a = [str(self.run_binary), "serve", "--host", self.host, "--port", str(self.port), "--name", self.served_name,
             "--context", str(self.context), "--max-tokens", str(self.max_tokens), "--parallel", str(self.parallel),
             "--keep-warm", str(self.keep_warm), "--no-update-check"]
        if self.temperature:
            a += ["--temperature", self.temperature]
        a.append("--thinking" if self.thinking else "--no-thinking")
        if self.thinking_budget > 0:
            a += ["--thinking-budget", str(self.thinking_budget)]
        if self.drafts:
            a += ["--drafter", str(self.drafter), "--drafter-bits", str(self.drafter_bits)]
        else:
            a.append("--no-drafts")
        if self.prompt_cache_gib >= 0:
            a += ["--prompt-cache-gib", f"{self.prompt_cache_gib:g}"]
            if self.prompt_cache_over_cap and self.prompt_cache_gib > 0:
                a.append("--prompt-cache-over-cap")
        if self.vision:
            a += ["--vision", "--vision-image-tokens", str(self.vision_image_tokens),
                  "--vision-max-images", str(self.vision_max_images)]
            if self.vision_urls:
                a.append("--vision-urls")
        if self.api_key:
            a += ["--api-key", self.api_key]
        a += self.extra_args.split()
        a.append(str(self.model))
        return a

    def server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["TENSORFOLD_LIBTURBOJPEG"] = str(self.jpeg_lib)
        return env

    def download_args(self) -> list[list[str]]:
        """The publisher's checkpoint into model_dir and the drafter into the Hugging Face cache, both at their pins."""
        return [["hf", "download", self.source_repo, "--revision", self.source_revision, "--local-dir", str(self.model)],
                ["hf", "download", self.drafter_repo, "--revision", self.drafter_revision]]


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
    for fld in fields(Settings):
        kind = type(getattr(s, fld.name))
        if fld.name in merged:
            setattr(s, fld.name, _coerce(kind, merged[fld.name]))
        if fld.name.upper() in env:
            setattr(s, fld.name, _coerce(kind, env[fld.name.upper()]))
    return s


def validate(s: Settings) -> list[str]:
    """The settings' problems, worded for the user (empty: none)."""
    out = []
    if not s.binary and not s.tensorfold_sha:
        out.append("no engine: set tensorfold_sha (local.toml) or binary")
    if not (s.model / "config.json").exists():
        out.append(f"no checkpoint at {s.model}: python3 -m mac download")
    if s.drafts and not (s.drafter / "config.json").exists():
        out.append(f"drafts = true but there is no drafter at {s.drafter}: hf download {s.drafter_repo} --revision {s.drafter_revision}")
    if s.drafter_bits not in (0, 4):
        out.append("drafter_bits must be 4 (prepared 4-bit copies) or 0 (bf16)")
    if s.parallel < 1 or s.parallel > 16:
        out.append("parallel must be 1 to 16 (measured up to 16; at 262k each stream takes 5.9 GiB, see README)")
    if s.context < 1 or s.context > 262144:
        out.append("context must be 1 to 262144 (the checkpoint's window; longer windows are not supported)")
    if not s.run_binary.exists():
        out.append(f"no engine binary at {s.run_binary}: python3 -m mac build")
    if s.vision and not s.jpeg_lib.exists():
        out.append(f"vision = true but there is no libjpeg-turbo at {s.jpeg_lib}: python3 -m mac build")
    return out
