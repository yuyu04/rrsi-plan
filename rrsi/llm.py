# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Claude client for the search roles via Vertex, Anthropic or Claude Code.

Vertex rotates over configured GCP projects; the Anthropic API uses one client.
Both cache stable prefixes as ephemeral content blocks. Claude Code uses a
subscription subprocess with the prefix concatenated into the prompt.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import tempfile
import threading
import time

MODEL = os.environ.get("RRSI_SEARCH_MODEL", "claude-opus-4-8")
_PROJECTS = [
    {"project_id": p.strip(), "region": os.environ.get("RRSI_VERTEX_REGION", "global")}
    for p in os.environ.get("RRSI_VERTEX_PROJECTS", "").split(",")
    if p.strip()
]
MAX_TOKENS = 20_000

_clients: dict = {}
_clients_lock = threading.Lock()
_cli_sem = None
_rr = itertools.count()
_JSON_SUFFIX = ("\n\nOutput ONLY a single valid JSON object. No prose before or "
                "after, no markdown fences.")


def _client_for(idx: int):
    from anthropic import AnthropicVertex
    if not _PROJECTS:
        raise RuntimeError("set RRSI_VERTEX_PROJECTS to a comma-separated list of GCP "
                           "projects with Claude on Vertex AI enabled")
    with _clients_lock:
        c = _clients.get(idx)
        if c is None:
            c = AnthropicVertex(**_PROJECTS[idx])
            _clients[idx] = c
        return c


def _backend() -> str:
    backend = os.environ.get("RRSI_LLM_BACKEND", "vertex").strip().lower()
    if backend not in {"vertex", "anthropic", "claude_cli"}:
        raise ValueError(f"invalid RRSI_LLM_BACKEND: {backend}")
    return backend


def _anthropic_client():
    from anthropic import Anthropic
    with _clients_lock:
        c = _clients.get("anthropic")
        if c is None:
            c = Anthropic()
            _clients["anthropic"] = c
        return c


def _api_generate(client, prompt, sys_prompt, mdl, max_tokens, cache_prefix):
    content = ([{"type": "text", "text": cache_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt}] if cache_prefix else prompt)
    kwargs = {"model": mdl, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": content}]}
    if sys_prompt:
        kwargs["system"] = sys_prompt
    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", "") == "text")


def _cli_env():
    # Keep subscription OAuth, but remove API auth and inherited CLI settings.
    excluded = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL", "CLAUDECODE"}
    return {k: v for k, v in os.environ.items()
            if k not in excluded and
            (not k.startswith("CLAUDE_CODE_") or k == "CLAUDE_CODE_OAUTH_TOKEN")}


def _cli_generate(prompt, sys_prompt, mdl, cache_prefix):
    global _cli_sem
    argv = [os.environ.get("RRSI_CLAUDE_BIN", "claude"), "-p",
            "--output-format", "json", "--model", mdl, "--tools", "",
            "--system-prompt", sys_prompt or "You are a helpful assistant.",
            "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", "--disable-slash-commands"]
    stdin = cache_prefix + "\n\n" + prompt if cache_prefix else prompt
    env = _cli_env()
    timeout = float(os.environ.get("RRSI_CLI_TIMEOUT", "600"))
    with _clients_lock:
        if _cli_sem is None:
            _cli_sem = threading.BoundedSemaphore(
                max(1, int(os.environ.get("RRSI_CLI_CONCURRENCY", "2"))))
        sem = _cli_sem
    with sem:
        resp = subprocess.run(argv, input=stdin, text=True, capture_output=True,
                              cwd=tempfile.gettempdir(), env=env, timeout=timeout)
    if resp.returncode != 0:
        raise RuntimeError(f"claude exited {resp.returncode}: {resp.stderr[-500:]}")
    data = json.loads(resp.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude error: {data.get('result')}")
    return data.get("result") or ""


def extract_json(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{") and not t.startswith("["):
        start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
        if start != -1:
            t = t[start:]
    if t and t[0] == "{" and not t.endswith("}"):
        end = t.rfind("}")
        if end != -1:
            t = t[:end + 1]
    if t and t[0] == "[" and not t.endswith("]"):
        end = t.rfind("]")
        if end != -1:
            t = t[:end + 1]
    return t


def generate(prompt: str, system: str | None = None, max_retries: int = 6,
             json_only: bool = False, model: str | None = None,
             max_tokens: int = MAX_TOKENS, cache_prefix: str | None = None) -> str:
    backend = _backend()
    mdl = model or MODEL
    sys_prompt = (system or "") + (_JSON_SUFFIX if json_only else "")
    n = len(_PROJECTS)
    start = next(_rr) if backend == "vertex" else 0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        if backend == "vertex":
            idx = (start + attempt) % n
        try:
            if backend == "claude_cli":
                # CLI has no per-call cap; the env cap errors instead of truncating.
                text = _cli_generate(prompt, sys_prompt, mdl, cache_prefix)
            else:
                client = _client_for(idx) if backend == "vertex" else _anthropic_client()
                text = _api_generate(client, prompt, sys_prompt, mdl,
                                     max_tokens, cache_prefix)
            if text:
                return extract_json(text) if json_only else text
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001 - retry backend failures
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"generate failed after {max_retries} tries: {last_err}")


if __name__ == "__main__":
    print(generate("Reply with exactly: OK", max_tokens=16))
