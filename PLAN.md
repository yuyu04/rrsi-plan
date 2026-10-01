# RRSI search-role LLM backends — implementation plan

Base: google-research/rrsi `be50316`.
Scope: `rrsi/llm.py`, new `tests/test_llm_backends.py`, `README.md`. Do NOT touch
`rrsi/analyst.py`, `critic.py`, `propose.py`, `digester.py`, `domains/`, or `pyproject.toml`.

## Context (already verified)

- All four roles call `rrsi.llm.generate(prompt, system=, json_only=, model=, max_tokens=, cache_prefix=)`
  (propose.py:311 passes `cache_prefix=stable`; the others pass prompt/system/json_only/model).
  Signature and return type (str) must stay identical.
- Python 3.14, pytest 9 available. `anthropic` is NOT installed system-wide → tests must
  inject a fake `anthropic` module via `sys.modules` (monkeypatch), never import the real one.
- Existing tests: `python3 -m pytest -q tests/` and `python3 tests/test_core.py` — both must keep passing.
- `claude` CLI 2.1.x flags verified by hand:
  - no `--max-turns`; `--tools ""` disables all built-in tools → single turn (`num_turns: 1`).
  - `--output-format json` → one JSON object on stdout with keys `result` (str), `is_error` (bool), `num_turns`, ...
  - `--system-prompt <text>` replaces the default Claude Code system prompt
    (CLI still appends a short "# Environment" block; acceptable).
  - `--setting-sources ""`, `--strict-mcp-config`, `--no-session-persistence`, `--disable-slash-commands`
    keep user hooks / MCP / skills / session files out.
  - Do NOT use `--bare` (forces ANTHROPIC_API_KEY auth, defeats subscription).
  - Do NOT set `CLAUDE_CODE_MAX_OUTPUT_TOKENS`: exceeding it is a hard error, not truncation.
    `max_tokens` is therefore ignored by the claude_cli backend (document it).

## 1. `rrsi/llm.py`

Keep the module's existing style (short helpers, terse comments). Update the module docstring
to describe the three backends.

- `_backend()` reads `RRSI_LLM_BACKEND` **at call time** (default `"vertex"`), lowercased/stripped;
  anything outside `{"vertex", "anthropic", "claude_cli"}` → `ValueError` (raised before any retry).
- Shared retry loop in `generate()` exactly like today: `for attempt in range(max_retries)`, call the
  backend; non-empty text → return (`extract_json(text)` if `json_only`); empty → `last_err = RuntimeError("empty response")`
  (no sleep, as today); exception → `last_err = e; time.sleep(min(2 ** attempt, 30))`.
  Final: `RuntimeError(f"generate failed after {max_retries} tries: {last_err}")`.
  `sys_prompt = (system or "") + (_JSON_SUFFIX if json_only else "")` computed once, as today.
- **vertex** (default): behavior byte-for-byte unchanged — round-robin `_rr` start, `idx = (start + attempt) % n`,
  `_client_for(idx)` with `AnthropicVertex`, content blocks with `cache_control: ephemeral` when `cache_prefix`.
- **anthropic**: `anthropic.Anthropic()` (reads ANTHROPIC_API_KEY), one cached client under `_clients_lock`
  (store under a distinct key, e.g. `_clients["anthropic"]`). Same `messages.create` kwargs and the same
  cache_control content blocks as vertex. Factor the kwargs/text extraction into one helper used by both.
- **claude_cli**:
  - argv: `[os.environ.get("RRSI_CLAUDE_BIN", "claude"), "-p", "--output-format", "json", "--model", mdl,
    "--tools", "", "--system-prompt", sys_prompt or "You are a helpful assistant.",
    "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"]`
    where `mdl = model or MODEL` (MODEL = `RRSI_SEARCH_MODEL`, unchanged).
  - stdin (`input=`, `text=True`): `cache_prefix + "\n\n" + prompt` if `cache_prefix` else `prompt`.
  - `cwd=tempfile.gettempdir()` (avoid picking up a project CLAUDE.md).
  - env = `_cli_env()`: copy of `os.environ` minus `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`,
    `ANTHROPIC_BASE_URL`, `CLAUDECODE`, and every `CLAUDE_CODE_*` key **except** `CLAUDE_CODE_OAUTH_TOKEN`
    (that one is how a headless subscription login is passed). Short comment why.
  - timeout: `float(os.environ.get("RRSI_CLI_TIMEOUT", "600"))` passed to `subprocess.run`;
    `TimeoutExpired` just propagates into the retry loop.
  - concurrency: module-level `threading.BoundedSemaphore`, created lazily on first CLI call with
    size `max(1, int(os.environ.get("RRSI_CLI_CONCURRENCY", "2")))` (global `_cli_sem`, guarded by `_clients_lock`).
    Hold it only around `subprocess.run` (not during backoff sleep).
  - parsing: returncode != 0 → `RuntimeError(f"claude exited {rc}: {stderr[-500:]}")`;
    `data = json.loads(stdout)`; `data.get("is_error")` → `RuntimeError(f"claude error: {data.get('result')}")`;
    return `data.get("result") or ""`.
  - `max_tokens` ignored (comment: CLI has no per-call cap; the env cap errors instead of truncating).
- `__main__` stays `print(generate("Reply with exactly: OK", max_tokens=16))`.

## 2. `tests/test_llm_backends.py` (new, pytest, no network)

Same license header as `tests/test_core.py`, same `sys.path.insert` idiom. Fixture that: inserts a fake
`anthropic` module (classes `Anthropic`, `AnthropicVertex` recording ctor kwargs and `messages.create` kwargs,
returning an object whose `.content` is `[SimpleNamespace(type="text", text=...)]`), patches
`rrsi.llm.time.sleep` to a no-op, resets `llm._clients` and `llm._cli_sem`, and clears the RRSI_* env vars.
Fake `subprocess.run` via `monkeypatch.setattr(llm.subprocess, "run", fake)`.

Cases:
1. default (unset) backend → AnthropicVertex used with `RRSI_VERTEX_PROJECTS`; kwargs unchanged incl. cache_control block (set `llm._PROJECTS` via monkeypatch).
2. `anthropic`: Anthropic (not Vertex) client; `cache_prefix` → 2 content blocks, first has `cache_control == {"type": "ephemeral"}`; `json_only` → system ends with `_JSON_SUFFIX` and fenced ```json``` output is extracted.
3. `claude_cli` argv: `-p`, `--output-format json`, `--tools ""` (adjacent pair), `--system-prompt` value == system + suffix, `--model` from `model=` arg, and from `RRSI_SEARCH_MODEL` default when no arg (monkeypatch `llm.MODEL`).
4. `claude_cli` stdin == `prefix + "\n\n" + prompt`; env lacks ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL/CLAUDECODE/CLAUDE_CODE_SESSION_ID but keeps CLAUDE_CODE_OAUTH_TOKEN and e.g. PATH; `timeout` == RRSI_CLI_TIMEOUT.
5. `claude_cli` json_only: result "Here:\n```json\n{\"a\": 1}\n```" → `'{"a": 1}'`.
6. retry: nonzero exit, then `is_error: true`, then `TimeoutExpired`, then success → returns success text after 4 calls; with max_retries=3 and all failing → RuntimeError mentioning "generate failed after 3 tries".
7. concurrency: `RRSI_CLI_CONCURRENCY=2`, fake run sleeps ~50ms and tracks peak in-flight; 6 threads → peak == 2.
8. invalid `RRSI_LLM_BACKEND=bogus` → ValueError, fake run never called.

## 3. README.md

After the paragraph at the "proposer, the analyst, the critic and the frozen policy are Claude Opus 4.8 on Vertex AI"
section (~line 72-78), add a short subsection "Search-role backend" with:
- table/list of `RRSI_LLM_BACKEND` = `vertex` (default) | `anthropic` (ANTHROPIC_API_KEY) | `claude_cli` (Claude Pro/Max/Team subscription via `claude -p`).
- claude_cli setup: install Claude Code, log in once (`claude` → `/login`, or `claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN` for headless), env vars `RRSI_SEARCH_MODEL`, `RRSI_CLI_CONCURRENCY` (default 2), `RRSI_CLI_TIMEOUT` (default 600s), `RRSI_CLAUDE_BIN`; smoke test `python3 -m rrsi.llm`.
- limitations: ANTHROPIC_API_KEY is stripped from the subprocess so usage bills to the subscription; no prompt caching (prefix concatenated); `max_tokens` not enforced; subscription rate limits apply → keep concurrency low.
- **Caution**: only the search roles move. The frozen policy model (`domains/*`, LiteLLM `policy_model` / `ORCHESTRATOR_MODEL`) is unchanged and still needs Vertex or an API key.

## 4. Verification (done by the planner, not the implementer)

- `python3 -m pytest -q tests/` and `python3 tests/test_core.py`
- `RRSI_LLM_BACKEND=claude_cli python3 -m rrsi.llm` (one real subscription call)
