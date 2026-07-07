"""User-level kbench configuration.

Precedence (highest to lowest):
    1. CLI flags (e.g. --remote-host)
    2. Environment variables (KBENCH_*)
    3. ~/.kbench.json
    4. src/config.py DEFAULT_REMOTE / DEFAULT_MIRRORS

The loader is stdlib-only (no pyyaml). Team members write JSON with $HOME
env var expansion supported for path-like values.

Example ~/.kbench.json:

    {
      "remote": {
        "ssh_user_host": "user@host",
        "ssh_port": 22,
        "workdir_root": "/root/kbench",
        "container_proxy": ""
      },
      "mirrors": {
        "gh_prefix": "https://ghfast.top/",
        "pypi_index": "https://pypi.tuna.tsinghua.edu.cn/simple/"
      },
      "keys": {
        "PPIO_API_KEY": "sk_..."
      }
    }

Populated by `./kbench init` (interactive). Mirrors are optional (default
PRC-friendly; set "mirrors": {} to disable).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

USER_CONFIG_PATH = Path.home() / ".kbench.json"


def load_user_config() -> dict[str, Any]:
    """Read ~/.kbench.json. Returns {} if missing or invalid.

    Missing file is normal (not an error) — the loader falls back to
    src/config.py DEFAULTS. Invalid JSON prints a warning but does not raise.
    """
    if not USER_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(USER_CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        import sys
        print(f"[kbench] warning: {USER_CONFIG_PATH} has invalid JSON: {e}",
              file=sys.stderr)
        return {}


def _expand_remote_path(path: str, ssh_user_host: str) -> str:
    """Expand path templates for shared-server workdirs.

    Supported placeholders:
      ~              -> /home/<remote-user>   (best-effort; fallback: bash $HOME)
      $USER / ${USER} -> local $USER (the person running kbench)
      $HOME          -> local $HOME
      {user}         -> local $USER (explicit template form)

    Rationale: on a shared test server, each teammate wants their own
    workdir under their own home. The remote user comes from the
    SSH spec's leftmost segment (e.g. 'gailun' in 'gailun@root@host@jump'),
    and 'root' is treated as an explicit shared account.

    Example:
        workdir_root = "~/kbench"  + ssh_user_host = "alice@host"
        -> "/home/alice/kbench"

        workdir_root = "/home/$USER/kbench" (local user is bob)
        -> "/home/bob/kbench"
    """
    if not path:
        return path
    # local user
    local_user = os.environ.get("USER", "")
    local_home = os.environ.get("HOME", "")
    result = path
    if "$USER" in result or "${USER}" in result:
        result = result.replace("${USER}", local_user).replace("$USER", local_user)
    if "$HOME" in result or "${HOME}" in result:
        result = result.replace("${HOME}", local_home).replace("$HOME", local_home)
    if "{user}" in result:
        result = result.replace("{user}", local_user)
    # remote user: leftmost segment of ssh_user_host before '@'
    if result.startswith("~"):
        remote_user = ""
        if ssh_user_host and "@" in ssh_user_host:
            remote_user = ssh_user_host.split("@", 1)[0]
        if remote_user and remote_user != "root":
            result = f"/home/{remote_user}" + result[1:]
        elif remote_user == "root":
            result = "/root" + result[1:]
        # else leave ~ literal — remote bash will expand it on the fly
    return result


def resolve_remote(cli_host: str | None = None,
                   cli_port: int | None = None,
                   cli_workdir: str | None = None) -> dict[str, Any]:
    """Merge remote config: CLI > env > ~/.kbench.json > src/config DEFAULT_REMOTE.

    Returns a dict with keys: ssh_user_host, ssh_port, workdir, container_proxy.
    workdir defaults to <workdir_root>/kimi-code-bench when only workdir_root
    is given, with $USER / ~ / {user} expansion for shared-server layouts.
    """
    from config import DEFAULT_REMOTE

    user = load_user_config().get("remote", {})
    result = dict(DEFAULT_REMOTE)

    # ~/.kbench.json overrides
    if user.get("ssh_user_host"):
        result["ssh_user_host"] = user["ssh_user_host"]
    if user.get("ssh_port"):
        result["ssh_port"] = int(user["ssh_port"])
    if user.get("workdir"):
        result["workdir"] = user["workdir"]
    elif user.get("workdir_root"):
        result["workdir"] = f"{user['workdir_root'].rstrip('/')}/kimi-code-bench"
    if user.get("container_proxy") is not None:
        # allow "" to explicitly disable proxy
        result["container_proxy"] = user["container_proxy"]

    # env overrides
    if os.environ.get("KBENCH_REMOTE_HOST"):
        result["ssh_user_host"] = os.environ["KBENCH_REMOTE_HOST"]
    if os.environ.get("KBENCH_REMOTE_PORT"):
        result["ssh_port"] = int(os.environ["KBENCH_REMOTE_PORT"])
    if os.environ.get("KBENCH_REMOTE_WORKDIR"):
        result["workdir"] = os.environ["KBENCH_REMOTE_WORKDIR"]

    # CLI wins
    if cli_host:
        result["ssh_user_host"] = cli_host
    if cli_port:
        result["ssh_port"] = int(cli_port)
    if cli_workdir:
        result["workdir"] = cli_workdir

    # Expand ~ / $USER / {user} placeholders (mostly for shared-server layouts)
    result["workdir"] = _expand_remote_path(result["workdir"], result["ssh_user_host"])

    return result


def resolve_mirrors() -> dict[str, str]:
    """Merge mirrors: env > ~/.kbench.json > src/config DEFAULT_MIRRORS.

    Set KBENCH_MIRRORS_OFF=1 (or `"mirrors": {}` explicitly {} in JSON) to
    disable mirrors entirely (empty gh_prefix, direct pypi).
    """
    from config import DEFAULT_MIRRORS

    if os.environ.get("KBENCH_MIRRORS_OFF") == "1":
        return {"gh_prefix": "", "pypi_index": "https://pypi.org/simple/"}

    user = load_user_config().get("mirrors")
    if user is not None:
        # empty dict = disable
        if not user:
            return {"gh_prefix": "", "pypi_index": "https://pypi.org/simple/"}
        result = dict(DEFAULT_MIRRORS)
        if "gh_prefix" in user:
            result["gh_prefix"] = user["gh_prefix"]
        if "pypi_index" in user:
            result["pypi_index"] = user["pypi_index"]
        return result

    return dict(DEFAULT_MIRRORS)


def resolve_api_key(key_ref: str) -> str:
    """Resolve an API key by reference:
    - starts with $ENV: FIRST try user config keys.ENV (unified single source),
        FALLBACK to env variable ENV
    - starts with ~ or $HOME: read from file
    - stored in ~/.kbench.json 'keys' section: look up by name
    - literal string: return as-is (rare, discouraged for keys)

    Rationale for $ENV-first-in-keys: `./kbench init` writes to
    ~/.kbench.json's keys section, and presets reference "$PPIO_API_KEY".
    We want both paths to work; keys.PPIO_API_KEY is preferred so users
    can rotate keys in one place without editing shell rc files.

    Examples:
        resolve_api_key("$PPIO_API_KEY")
            -> ~/.kbench.json.keys.PPIO_API_KEY if set,
               else os.environ["PPIO_API_KEY"]
        resolve_api_key("PPIO_API_KEY")       -> user config keys.PPIO_API_KEY
        resolve_api_key("~/.secrets/ppio")    -> file content
        resolve_api_key("sk_...")             -> literal
    """
    if not key_ref:
        return ""
    if key_ref.startswith("~") or key_ref.startswith("$HOME"):
        path = Path(os.path.expandvars(os.path.expanduser(key_ref)))
        if path.exists():
            return path.read_text().strip()
        return ""
    if key_ref.startswith("$"):
        name = key_ref[1:]
        user_keys = load_user_config().get("keys", {})
        if name in user_keys:
            return user_keys[name]
        val = os.environ.get(name, "")
        if not val:
            import sys
            print(f"[kbench] warning: {key_ref} not in ~/.kbench.json keys "
                  f"nor env vars", file=sys.stderr)
        return val
    # Try user config keys section (no $ prefix)
    user_keys = load_user_config().get("keys", {})
    if key_ref in user_keys:
        return user_keys[key_ref]
    # Literal
    return key_ref
