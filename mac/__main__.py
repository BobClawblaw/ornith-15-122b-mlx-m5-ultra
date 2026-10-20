"""python3 -m mac venv|convert|up|down|status|logs|command|chat|validate|launchd"""

import argparse
import shlex
import shutil
import subprocess
import sys

from . import config, launchd, serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m mac", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("venv", help="the serving stack (requirements.txt) into .venv with uv")
    c = sub.add_parser("convert", help="Ornith's release (in the Hugging Face cache) to the 4- or 8-bit checkpoint")
    c.add_argument("--bits", type=int, default=None, help="4 or 8 (default: the settings' bits)")
    up = sub.add_parser("up", help="start the server (in the background; --foreground for the launchd unit)")
    up.add_argument("--foreground", action="store_true")
    sub.add_parser("down", help="stop the server `up` started")
    sub.add_parser("status", help="pid, url and the server's health")
    logs = sub.add_parser("logs", help="the server log's last lines")
    logs.add_argument("-n", type=int, default=40)
    sub.add_parser("command", help="print the serve command")
    ch = sub.add_parser("chat", help="one prompt to the running server")
    ch.add_argument("prompt")
    ch.add_argument("--max-tokens", type=int, default=128)
    sub.add_parser("validate", help="check the settings, the checkpoint and the venv without starting anything")
    ld = sub.add_parser("launchd", help="write the launchd unit (not loaded) serving PORT to the LAN")
    ld.add_argument("--port", type=int, default=8888)
    a = p.parse_args(argv)
    s = config.load()
    if a.cmd == "venv":
        root = str(config.ROOT)
        for cmd in (["uv", "venv", "--python", "3.12", f"{root}/.venv"],
                    ["uv", "pip", "install", "--python", f"{root}/.venv/bin/python", "-r", f"{root}/requirements.txt"]):
            print("+", shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        # the shim mlx-vlm's recurrent layers need on vllm-mlx's caches (tools/sitecustomize.py says why)
        site = subprocess.run([f"{root}/.venv/bin/python", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                              capture_output=True, text=True, check=True).stdout.strip()
        shutil.copy(config.ROOT / "tools" / "sitecustomize.py", site)
        print(f"+ tools/sitecustomize.py -> {site}")
    elif a.cmd == "convert":
        if not (s.source_snapshot / "model.safetensors.index.json").exists():
            raise SystemExit(f"{s.source_repo}@{s.source_revision[:8]} is not in the Hugging Face cache: "
                             f"hf download {s.source_repo} --revision {s.source_revision}")
        cmd, out = s.convert_args(a.bits or s.bits)
        print("+", shlex.join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=config.ROOT).returncode
        if rc == 0:
            print(f"wrote {out} and {out}-mtp")
        return rc
    elif a.cmd == "up":
        problems = config.validate(s)
        if problems:
            raise SystemExit("\n".join(problems))
        serve.up(s, foreground=a.foreground)
    elif a.cmd == "down":
        serve.down(s)
    elif a.cmd == "status":
        print(serve.status(s))
    elif a.cmd == "logs":
        print("\n".join(s.log_file.read_text(errors="replace").splitlines()[-a.n:]) if s.log_file.exists() else "no log")
    elif a.cmd == "command":
        print(shlex.join(s.serve_args()))
    elif a.cmd == "chat":
        print(serve.chat(s, a.prompt, a.max_tokens))
    elif a.cmd == "validate":
        problems = config.validate(s)
        print("\n".join(problems) if problems else f"ok: {s.served_name} from {s.model} ({s.bits}-bit) at {s.url}")
        return 1 if problems else 0
    elif a.cmd == "launchd":
        print(launchd.write(s, a.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
