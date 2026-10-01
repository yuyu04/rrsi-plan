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
"""Search-role backend tests: no API or Claude subprocess."""

import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrsi import llm  # noqa: E402


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    state = SimpleNamespace(constructors=[], calls=[], text="OK")

    class Anthropic:
        def __init__(self, **kwargs):
            state.constructors.append((type(self).__name__, kwargs))
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            state.calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=state.text)])

    class AnthropicVertex(Anthropic):
        pass

    module = ModuleType("anthropic")
    module.Anthropic = Anthropic
    module.AnthropicVertex = AnthropicVertex
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "_cli_sem", None)
    monkeypatch.setattr(llm, "_rr", itertools.count())
    for key in list(os.environ):
        if key.startswith("RRSI_"):
            monkeypatch.delenv(key)
    return state


def _result(text="OK", is_error=False, returncode=0, stderr=""):
    return SimpleNamespace(returncode=returncode, stderr=stderr,
                           stdout=json.dumps({"result": text, "is_error": is_error}))


def test_default_vertex(monkeypatch, fake_sdk):
    monkeypatch.setenv("RRSI_VERTEX_PROJECTS", "project-a,project-b")
    projects = [{"project_id": p, "region": "global"}
                for p in os.environ["RRSI_VERTEX_PROJECTS"].split(",")]
    monkeypatch.setattr(llm, "_PROJECTS", projects)
    assert llm.generate("prompt", system="system", model="model", max_tokens=123,
                        cache_prefix="prefix") == "OK"
    assert fake_sdk.constructors == [("AnthropicVertex", projects[0])]
    assert fake_sdk.calls == [{
        "model": "model", "max_tokens": 123, "system": "system",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "prompt"}]}]}]
    assert llm.generate("next") == "OK"
    assert fake_sdk.constructors[-1] == ("AnthropicVertex", projects[1])
    assert fake_sdk.calls[-1] == {
        "model": llm.MODEL, "max_tokens": llm.MAX_TOKENS,
        "messages": [{"role": "user", "content": "next"}]}


def test_anthropic(monkeypatch, fake_sdk):
    monkeypatch.setenv("RRSI_LLM_BACKEND", " Anthropic ")
    fake_sdk.text = '```json\n{"a": 1}\n```'
    assert llm.generate("prompt", system="system", json_only=True,
                        cache_prefix="prefix") == '{"a": 1}'
    assert fake_sdk.constructors == [("Anthropic", {})]
    kwargs = fake_sdk.calls[0]
    assert kwargs["system"] == "system" + llm._JSON_SUFFIX
    assert kwargs["model"] == llm.MODEL
    assert kwargs["max_tokens"] == llm.MAX_TOKENS
    assert kwargs["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "prompt"}]}]
    llm.generate("again")
    assert fake_sdk.constructors == [("Anthropic", {})]


@pytest.mark.parametrize("model", ["explicit-model", None])
def test_cli_argv(monkeypatch, model):
    monkeypatch.setenv("RRSI_LLM_BACKEND", "claude_cli")
    monkeypatch.setenv("RRSI_CLAUDE_BIN", "/fake/claude")
    monkeypatch.setenv("RRSI_SEARCH_MODEL", "default-model")
    monkeypatch.setattr(llm, "MODEL", os.environ["RRSI_SEARCH_MODEL"])
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        return _result('{"a": 1}')

    monkeypatch.setattr(llm.subprocess, "run", fake)
    assert llm.generate("prompt", system="system", json_only=True, model=model) == '{"a": 1}'
    argv, kwargs = calls[0]
    assert argv == ["/fake/claude", "-p", "--output-format", "json", "--model",
                    model or "default-model", "--tools", "", "--system-prompt",
                    "system" + llm._JSON_SUFFIX, "--setting-sources", "",
                    "--strict-mcp-config", "--no-session-persistence",
                    "--disable-slash-commands"]
    assert kwargs["input"] == "prompt"
    assert kwargs["text"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["cwd"] == tempfile.gettempdir()
    assert kwargs["timeout"] == 600.0


def test_cli_input_env_timeout(monkeypatch):
    monkeypatch.setenv("RRSI_LLM_BACKEND", "claude_cli")
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_MAX_OUTPUT_TOKENS"):
        monkeypatch.setenv(key, "fake-value")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-oauth")
    monkeypatch.setenv("PATH", "/fake/path")
    monkeypatch.setenv("RRSI_CLI_TIMEOUT", "12.5")

    def fake(argv, **kwargs):
        assert kwargs["input"] == "prefix\n\nprompt"
        assert kwargs["timeout"] == 12.5
        env = kwargs["env"]
        assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                    "CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_MAX_OUTPUT_TOKENS"} & env.keys()
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "fake-oauth"
        assert env["PATH"] == "/fake/path"
        assert argv[argv.index("--system-prompt") + 1] == "You are a helpful assistant."
        return _result()

    monkeypatch.setattr(llm.subprocess, "run", fake)
    assert llm.generate("prompt", cache_prefix="prefix", max_tokens=1) == "OK"
    assert os.environ["ANTHROPIC_API_KEY"] == "fake-value"


def test_cli_json_only(monkeypatch):
    monkeypatch.setenv("RRSI_LLM_BACKEND", "claude_cli")
    monkeypatch.setattr(llm.subprocess, "run",
                        lambda *a, **kw: _result('Here:\n```json\n{"a": 1}\n```'))
    assert llm.generate("prompt", json_only=True) == '{"a": 1}'


def test_cli_retry(monkeypatch):
    monkeypatch.setenv("RRSI_LLM_BACKEND", "claude_cli")
    responses = [_result(returncode=1, stderr="failure"),
                 _result("failure", is_error=True),
                 subprocess.TimeoutExpired("fake-claude", 600), _result("success")]
    calls = []

    def fake(*args, **kwargs):
        response = responses[len(calls)]
        calls.append(kwargs)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(llm.subprocess, "run", fake)
    assert llm.generate("prompt", max_retries=4) == "success"
    assert len(calls) == 4
    calls.clear()
    with pytest.raises(RuntimeError, match="generate failed after 3 tries"):
        llm.generate("prompt", max_retries=3)
    assert len(calls) == 3


def test_cli_concurrency(monkeypatch):
    monkeypatch.setenv("RRSI_LLM_BACKEND", "claude_cli")
    monkeypatch.setenv("RRSI_CLI_CONCURRENCY", "2")
    lock = threading.Lock()
    active = peak = 0

    def fake(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        threading.Event().wait(0.05)
        with lock:
            active -= 1
        return _result()

    monkeypatch.setattr(llm.subprocess, "run", fake)
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(llm.generate, ["prompt"] * 6)) == ["OK"] * 6
    assert peak == 2


def test_invalid_backend(monkeypatch, fake_sdk):
    calls = []
    monkeypatch.setattr(llm.subprocess, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setenv("RRSI_LLM_BACKEND", "bogus")
    with pytest.raises(ValueError, match="invalid RRSI_LLM_BACKEND"):
        llm.generate("prompt")
    assert calls == []
    assert fake_sdk.constructors == []
