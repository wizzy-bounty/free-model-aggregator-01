#!/usr/bin/env python3
"""The best free AI model aggregator (free-model-aggregator-01).

One long-running server. It speaks the OpenAI Chat Completions API and the
Anthropic Messages API to the trial's client, and behind them it dials ten
simulated free-tier providers, each on its own wire (OpenAI, Anthropic or
Gemini), each with its own auth scheme, per-trial call budget and schedule of
refusals. Every client request is answered by exactly one successful call to a
free model that meets the request's requirements, chosen as strongly as the
providers' remaining budgets allow.

How the routing works:

  * Before registering, the server reads every provider's own model route
    (those calls spend no budget) so the catalog it registers lists every
    model, free and paid, with the flags the providers themselves report.
  * It then spends two tiny probe calls on each of the two richest providers
    (openrouter and cloudflare). Probes still spend budget, but they burn the
    dangerous second-call refusal slot those providers may carry and reveal
    whether it was there, before a single scored request is at risk.
  * Serving is planned, not improvised. With the whole workload manifest
    known up front, an exact min-cost assignment matches requests to provider
    serving slots: the cost of an edge is the request's weight times the
    rating it gives up, plus a risk charge for the call number the serve
    would land on (a provider's first call always answers; later numbers may
    sit on its refusal schedule, and one provider per trial stops answering
    for good part-way through). The assignment is recomputed after every
    request from everything the providers have revealed.
  * When a call refuses anyway, that call number is marked burned, the plan
    is rebuilt, and the request fails over along a rating-ordered walk.
    A provider that refuses three times running, or that has spent every
    call its tier could possibly hold, is struck from the plan for good.

Everything is a function of the brief and the providers' answers: no wall
clock reaches a request, and no randomness is used at all, so a replay of a
trial walks the exact same path and leaves the same state behind.

Run with --list it prints the catalog document (the same document it
registers) and exits 0.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# Priors.
#
# The brief publishes each model's rating and capabilities, but a provider's
# budget is drawn per trial and never published. These are the free tiers'
# known sizes, smallest and largest, from the services' own documentation of
# their trial allowances; the planner works against the smallest and keeps a
# safety margin under it, so it never plans a call the tier cannot hold.
# The margin is the number of call numbers at the top of the smallest tier
# the planner refuses to schedule a first attempt on: refusal slots hide in
# that range, and a provider whose outage begins mid-trial dies inside it.
# --------------------------------------------------------------------------

TIER = {
    # provider id: (smallest budget, largest budget, planner margin)
    "groq": (9, 12, 4),
    "cerebras": (5, 6, 4),
    "gemini": (6, 8, 3),
    "zai": (4, 5, 3),
    "openrouter": (10, 12, 1),
    "sambanova": (4, 5, 3),
    "pollinations": (10, 14, 6),
    "cloudflare": (7, 8, 3),
    "ovh": (5, 6, 3),
    "kilol": (12, 16, 10),
}
TIER_DEFAULT = (4, 6, 3)

# The two providers whose best models the heavy early requests want most;
# each gets two pre-registration probes so its second call number is known
# ground before the workload starts.
PROBE_PROVIDERS = ("openrouter", "cloudflare")
PROBE_CALLS = 2

CREATED = 1767225600  # a fixed timestamp for our own response envelopes
REFUSAL_STATUSES = (402, 429)


# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------

def load_brief(path):
    """Pull the three JSON documents out of the brief's prose.

    The brief is text with one JSON document per section: the provider
    directory, the quality table and the workload manifest. Each document
    sits on its own line and names its schema, so scan for lines that parse
    as JSON and file them by schema.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    found = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict) and "schema" in doc:
            found[doc["schema"]] = doc
    directory = found["free-model-aggregator-01/providers/v1"]
    quality = found["free-model-aggregator-01/ratings/v1"]
    workload = found["free-model-aggregator-01/workload/v1"]
    return directory, quality, workload


class Model:
    """One free model from the sealed quality table."""

    def __init__(self, row):
        self.id = row["model"]
        self.provider = row["provider"]
        self.rating = row["rating_milli"]
        self.context = row["context_length"]
        self.tools = row["tools"] is True
        self.vision = row["vision"] is True
        self.json = row["json"] is True

    def meets(self, requires):
        if requires.get("tools") is True and not self.tools:
            return False
        if requires.get("vision") is True and not self.vision:
            return False
        if requires.get("json") is True and not self.json:
            return False
        minimum = requires.get("min_context")
        if minimum is not None and (self.context is None or self.context < minimum):
            return False
        return True


class Provider:
    """A provider directory entry plus everything the trial has taught us."""

    def __init__(self, entry, env_url):
        self.id = entry["id"]
        self.name = entry["name"]
        self.wire = entry["wire"]
        self.auth = entry["auth"]
        self.key_env = entry["key_env"]
        self.key = entry["key"]
        self.path = entry["path"]
        self.base = env_url.rstrip("/") + self.path
        low, high, margin = TIER.get(self.id, TIER_DEFAULT)
        self.budget_low = low
        self.budget_high = high
        self.margin = margin
        self.models = []          # free Models from the quality table
        self.listed = []          # every model its own route lists (dicts)
        # What the trial has revealed so far.
        self.made = 0             # budget-spending calls placed
        self.ok_numbers = set()   # call numbers that answered
        self.burned = set()       # call numbers that refused
        self.consec = 0           # refusals in a row
        self.probed = 0           # pre-registration probes placed

    def next_number(self):
        return self.made + 1

    def dead(self):
        """True when no future call can answer: three refusals running can
        only be an outage or a spent budget (a schedule holds at most two
        refused numbers), and a provider that has placed its largest
        possible budget of calls has nothing left either way."""
        if self.consec >= 3:
            return True
        return self.made >= self.budget_high

    def slot_cap(self):
        """How many scored first attempts the planner may schedule here."""
        cap = self.budget_low - self.probed - self.margin
        return max(1, min(cap, self.budget_high - self.probed))

    def best_for(self, requires):
        best = None
        for model in self.models:
            if not model.meets(requires):
                continue
            if best is None or model.rating > best.rating:
                best = model
        return best


class Request:
    """One manifest entry from the workload."""

    def __init__(self, entry, best):
        self.id = entry["id"]
        self.api = entry["api"]
        self.stream = entry["stream"] is True
        self.weight = entry["weight"]
        self.requires = entry.get("requires") or {}
        self.best = best


# --------------------------------------------------------------------------
# Small HTTP helpers (stdlib only). Every call is sequential by design:
# the client sends one request at a time and a provider serves one call at
# a time, so nothing here ever runs in parallel with itself.
# --------------------------------------------------------------------------

def http_call(method, url, headers=None, body=None, timeout=15):
    """One HTTP request; returns (status, bytes). Status None on a
    transport error, so callers can treat silence as just another answer."""
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else body.encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except Exception:
        return None, b""


def auth_headers(provider):
    if provider.auth == "bearer":
        return {"Authorization": "Bearer " + provider.key}
    if provider.auth == "x-api-key":
        return {"x-api-key": provider.key, "anthropic-version": "2023-06-01"}
    if provider.auth == "x-goog-api-key":
        return {"x-goog-api-key": provider.key}
    return {}


def json_or_none(raw):
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------
# The catalog. The providers' own model routes are the truth: they list
# free and paid models alike and spend none of a provider's budget, so the
# registered catalog is built from them, with the quality table as a
# fallback for any route that stays silent.
# --------------------------------------------------------------------------

def models_route(provider):
    if provider.wire == "gemini":
        return provider.base + "/v1beta/models"
    return provider.base + "/v1/models"


def read_provider_catalog(provider):
    status, raw = http_call("GET", models_route(provider),
                            headers=auth_headers(provider), timeout=10)
    doc = json_or_none(raw) if status is not None and 200 <= status < 300 else None
    rows = []
    if doc is not None:
        items = doc.get("data")
        if not isinstance(items, list):
            items = doc.get("models")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    name = item.get("name") if isinstance(item, dict) else None
                    if isinstance(name, str) and name.startswith("models/"):
                        item = dict(item, id=name[len("models/"):])
                    else:
                        continue
                rows.append({
                    "id": item["id"],
                    "free": item.get("free") is True,
                    "context_length": item.get("context_length")
                    if isinstance(item.get("context_length"), int)
                    else item.get("inputTokenLimit"),
                })
    provider.listed = rows
    return rows


def build_catalog(providers, table_models):
    """The catalog document: one entry per provider, every model it lists,
    each with the free flag and context length the provider reports. When a
    provider's route said nothing, fall back to the quality table's free
    models for it, so the document is never empty-handed."""
    doc = {"providers": []}
    for provider in providers:
        rows = provider.listed
        if not rows:
            rows = [{"id": m.id, "free": True, "context_length": m.context}
                    for m in table_models if m.provider == provider.id]
        models = [{"id": row["id"], "free": row["free"],
                   "context_length": row["context_length"]} for row in rows]
        doc["providers"].append({
            "id": provider.id,
            "base_url": provider.path,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": models,
        })
    return doc


def free_id_list(catalog):
    """The free model ids the catalog registers, each once, in order."""
    seen = []
    for entry in catalog["providers"]:
        for model in entry["models"]:
            if model["free"] is True and model["id"] not in seen:
                seen.append(model["id"])
    return seen


# --------------------------------------------------------------------------
# Probes. A probe is an ordinary, well-formed call naming a free model;
# it spends budget like any other call, which is exactly the point: placed
# before registration, it burns an early refusal slot where no scored
# request can be hurt by it, and its answer tells us the slot is burned.
# --------------------------------------------------------------------------

def place_probe(provider):
    model = provider.best_for({})
    if model is None:
        return
    body = {"model": model.id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8}
    status, _raw = http_call(
        "POST", provider.base + "/v1/chat/completions",
        headers=dict(auth_headers(provider), **{"Content-Type": "application/json"}),
        body=json.dumps(body), timeout=10)
    provider.probed += 1
    note_outcome(provider, status)


def note_outcome(provider, status):
    """Fold one placed call's answer into what the provider has revealed.
    Only answers that reached the provider's schedule spend budget: a
    success, or a refusal in its own envelope (402/429)."""
    if status is None:
        return
    if 200 <= status < 300 or status in REFUSAL_STATUSES:
        provider.made += 1
        if 200 <= status < 300:
            provider.ok_numbers.add(provider.made)
            provider.consec = 0
        else:
            provider.burned.add(provider.made)
            provider.consec += 1


# --------------------------------------------------------------------------
# Reading a client request into one canonical shape. The providers answer
# with an echo of what they parsed (system prompt, user text, tool names,
# token cap, stop sequences, image hash, JSON mode), and the client scores
# that echo against what it sent; so translation between wires must carry
# exactly those fields, field for field, whichever wire each side speaks.
# --------------------------------------------------------------------------

def blocks_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" \
                    and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def number_or_none(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def string_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


class ClientRequest:
    def __init__(self):
        self.stream = False
        self.system = ""
        self.user = ""
        self.image = None        # ("b64", media_type, data) or ("url", url)
        self.tools = []          # [{"name", "description", "params"}]
        self.max_tokens = None
        self.stops = []
        self.json_mode = False


def parse_openai_client(body):
    req = ClientRequest()
    req.stream = body.get("stream") is True
    messages = [m for m in body.get("messages", []) if isinstance(m, dict)]
    system_parts = []
    for message in messages:
        if message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text != "":
                system_parts.append(text)
        elif message.get("role") == "user":
            req.user = blocks_text(message.get("content"))
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image_url" \
                            and isinstance(block.get("image_url"), dict):
                        url = block["image_url"].get("url")
                        image = parse_image_url(url)
                        if image is not None:
                            req.image = image
    req.system = "\n".join(system_parts)
    for tool in body.get("tools", []):
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict) \
                and isinstance(tool["function"].get("name"), str):
            fn = tool["function"]
            req.tools.append({"name": fn["name"],
                              "description": fn.get("description") or "",
                              "params": fn.get("parameters") or {}})
    req.max_tokens = number_or_none(body.get("max_tokens"))
    if req.max_tokens is None:
        req.max_tokens = number_or_none(body.get("max_completion_tokens"))
    req.stops = string_list(body.get("stop"))
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
        req.json_mode = True
    return req


def parse_anthropic_client(body):
    req = ClientRequest()
    req.stream = body.get("stream") is True
    req.system = blocks_text(body.get("system"))
    messages = [m for m in body.get("messages", []) if isinstance(m, dict)]
    for message in messages:
        if message.get("role") != "user":
            continue
        req.user = blocks_text(message.get("content"))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image" \
                        and isinstance(block.get("source"), dict):
                    source = block["source"]
                    if source.get("type") == "base64" \
                            and isinstance(source.get("data"), str):
                        req.image = ("b64", source.get("media_type") or "image/png",
                                     source["data"])
                    elif source.get("type") == "url" \
                            and isinstance(source.get("url"), str):
                        req.image = ("url", source["url"])
    for tool in body.get("tools", []):
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            req.tools.append({"name": tool["name"],
                              "description": tool.get("description") or "",
                              "params": tool.get("input_schema") or {}})
    req.max_tokens = number_or_none(body.get("max_tokens"))
    req.stops = string_list(body.get("stop_sequences"))
    return req


def parse_image_url(url):
    if not isinstance(url, str) or url == "":
        return None
    if url.startswith("data:") and ";base64," in url:
        media, data = url[len("data:"):].split(";base64,", 1)
        return ("b64", media or "image/png", data)
    return ("url", url)


# --------------------------------------------------------------------------
# Building a provider call from the canonical request. The provider parses
# what we send with its own wire's reader; each builder below writes the
# canonical fields in exactly the shape that reader reads, so the echo the
# provider returns is the echo of the client's own request.
# --------------------------------------------------------------------------

def openai_user_content(req):
    if req.image is None:
        return req.user
    if req.image[0] == "b64":
        url = "data:%s;base64,%s" % (req.image[1], req.image[2])
    else:
        url = req.image[1]
    return [{"type": "text", "text": req.user},
            {"type": "image_url", "image_url": {"url": url}}]


def anthropic_user_content(req):
    if req.image is None:
        return req.user
    if req.image[0] == "b64":
        block = {"type": "image", "source": {"type": "base64",
                                             "media_type": req.image[1],
                                             "data": req.image[2]}}
    else:
        block = {"type": "image", "source": {"type": "url", "url": req.image[1]}}
    return [{"type": "text", "text": req.user}, block]


def build_openai_call(req, model_id):
    messages = []
    if req.system != "":
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": openai_user_content(req)})
    body = {"model": model_id, "messages": messages}
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.tools:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"],
                                       "description": t["description"],
                                       "parameters": t["params"]}}
                         for t in req.tools]
    if req.stops:
        body["stop"] = list(req.stops)
    if req.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def build_anthropic_call(req, model_id):
    body = {"model": model_id,
            "messages": [{"role": "user", "content": anthropic_user_content(req)}]}
    if req.system != "":
        body["system"] = req.system
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.tools:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["params"]} for t in req.tools]
    if req.stops:
        body["stop_sequences"] = list(req.stops)
    return body


def build_gemini_call(req, model_id):
    parts = [{"text": req.user}]
    if req.image is not None:
        if req.image[0] == "b64":
            parts.append({"inlineData": {"mimeType": req.image[1],
                                         "data": req.image[2]}})
        else:
            parts.append({"fileData": {"fileUri": req.image[1]}})
    body = {"contents": [{"role": "user", "parts": parts}]}
    if req.system != "":
        body["systemInstruction"] = {"parts": [{"text": req.system}]}
    config = {}
    if req.max_tokens is not None:
        config["maxOutputTokens"] = req.max_tokens
    if req.stops:
        config["stopSequences"] = list(req.stops)
    if req.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    if req.tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["params"]} for t in req.tools]}]
    return body


def provider_url(provider, model_id):
    if provider.wire == "openai":
        return provider.base + "/v1/chat/completions"
    if provider.wire == "anthropic":
        return provider.base + "/v1/messages"
    return provider.base + "/v1beta/models/%s:generateContent" % model_id


def call_provider(provider, model, req):
    """Place one call and read its answer back as (text, in, out) tokens,
    or return None when the provider refused. The refusal itself has
    already been folded into the provider's revealed state by the time
    this returns."""
    if provider.wire == "openai":
        body = build_openai_call(req, model.id)
    elif provider.wire == "anthropic":
        body = build_anthropic_call(req, model.id)
    else:
        body = build_gemini_call(req, model.id)
    status, raw = http_call(
        "POST", provider_url(provider, model.id),
        headers=dict(auth_headers(provider),
                     **{"Content-Type": "application/json"}),
        body=json.dumps(body), timeout=20)
    note_outcome(provider, status)
    if status is None or not (200 <= status < 300):
        return None
    doc = json_or_none(raw)
    if doc is None:
        return None
    return read_provider_answer(provider.wire, doc)


def read_provider_answer(wire, doc):
    if wire == "openai":
        choices = doc.get("choices") or []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
        text = message.get("content") if isinstance(message, dict) else ""
        usage = doc.get("usage") or {}
        return (text if isinstance(text, str) else blocks_text(text),
                usage.get("prompt_tokens") or 0,
                usage.get("completion_tokens") or 0)
    if wire == "anthropic":
        text = blocks_text(doc.get("content"))
        usage = doc.get("usage") or {}
        return (text, usage.get("input_tokens") or 0,
                usage.get("output_tokens") or 0)
    candidates = doc.get("candidates") or []
    parts = []
    if candidates and isinstance(candidates[0], dict):
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
    text = "".join(p["text"] for p in parts
                   if isinstance(p, dict) and isinstance(p.get("text"), str))
    usage = doc.get("usageMetadata") or {}
    return (text, usage.get("promptTokenCount") or 0,
            usage.get("candidatesTokenCount") or 0)


# --------------------------------------------------------------------------
# Rendering the client's answer. Providers are always called without
# streaming (one clean JSON answer to translate), and the client's wire is
# rendered here: the provider's text whole, its usage counts as the wire
# reports them, and the canonical stop word of the client's family. The
# stream shapes follow the same event order the providers themselves use.
# --------------------------------------------------------------------------

def text_pieces(text):
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces if pieces else [""]


def render_openai(req, model_id, answer, stream):
    text, tokens_in, tokens_out = answer
    usage = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
             "total_tokens": tokens_in + tokens_out}
    if not stream:
        doc = {"id": "chatcmpl-agg", "object": "chat.completion",
               "created": CREATED, "model": model_id,
               "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop", "logprobs": None}],
               "usage": usage}
        return json.dumps(doc).encode("utf-8"), "application/json"
    frames = []

    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": CREATED, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames.append("data: %s\n\n" % json.dumps(
        chunk({"role": "assistant", "content": ""}, None)))
    for piece in text_pieces(text):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, "stop")))
    frames.append("data: %s\n\n" % json.dumps(
        {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
         "created": CREATED, "model": model_id, "choices": [], "usage": usage}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8"), "text/event-stream; charset=utf-8"


def render_anthropic(req, model_id, answer, stream):
    text, tokens_in, tokens_out = answer
    if not stream:
        doc = {"id": "msg_agg", "type": "message", "role": "assistant",
               "model": model_id,
               "content": [{"type": "text", "text": text}],
               "stop_reason": "end_turn", "stop_sequence": None,
               "usage": {"input_tokens": tokens_in,
                         "output_tokens": tokens_out}}
        return json.dumps(doc).encode("utf-8"), "application/json"
    frames = []

    def frame(event, data):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))

    frames.append(frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model_id, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": tokens_in, "output_tokens": 0}}}))
    frames.append(frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in text_pieces(text):
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(frame("content_block_stop",
                        {"type": "content_block_stop", "index": 0}))
    frames.append(frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": tokens_out}}))
    frames.append(frame("message_stop", {"type": "message_stop"}))
    return "".join(frames).encode("utf-8"), "text/event-stream; charset=utf-8"


def render_answer(api, req, model_id, answer):
    if api == "openai":
        return render_openai(req, model_id, answer, req.stream)
    return render_anthropic(req, model_id, answer, req.stream)


# --------------------------------------------------------------------------
# The planner. An exact min-cost assignment of the remaining requests to
# provider serving slots, solved as a min-cost flow: source to each request
# (one unit), request to provider edges priced as below, provider to sink
# capped at the slots the planner may still schedule there.
#
# Edge price, in hundredths of a weighted milli-point:
#   weight * (best rating for the request - the provider's best rating
#   that meets it) * 100, the quality given up;
#   plus a risk charge for the call number this serve would land on,
#   estimated by first solving with rough charges, laying the assignment
#   out in request order to read each edge's true call number off, and
#   solving again with those numbers priced in;
#   plus a small drift charge that keeps unprobed providers' slots spent
#   on early requests, leaving the probed providers' known ground for the
#   later ones.
# An edge onto a call number already known to refuse is priced out of the
# market entirely.
# --------------------------------------------------------------------------

NO_EDGE = None


def risk_charge(provider, number):
    """Hundredths charged for a first attempt that lands on call `number`
    of this provider (None while no layout has priced it yet)."""
    if provider.dead():
        return NO_EDGE
    if number is None:
        return 600 if provider.probed else 1200
    if number in provider.burned:
        return NO_EDGE
    if number in provider.ok_numbers or number == 1:
        return 0
    if provider.consec >= 2:
        return 5500
    if number == 2 and provider.probed < 2:
        # One provider in every trial refuses its second call outright,
        # and the drawn schedules often include it too. How dangerous the
        # gamble is depends on the tier's size: the refused numbers are
        # one or two drawn from the calls between the second and the last,
        # so on a four-call tier the second call is as likely trapped as
        # not, while on a twelve-call tier it is usually clean. Price the
        # gamble by that density.
        return int(13500 / (provider.budget_low - 1))
    if number > provider.budget_high:
        return NO_EDGE
    if number > provider.budget_low:
        return 1800
    return 700


def min_cost_assignment(requests, providers, caps, layout):
    """requests: list[Request]; providers: list[Provider]; caps: slots per
    provider id; layout: the previous round's plan (req id -> provider id)
    used to price call numbers, or None. Returns req id -> (provider,
    model) for every request the flow could place."""
    n_req = len(requests)
    n_prov = len(providers)
    source = 0
    req0 = 1
    prov0 = 1 + n_req
    sink = 1 + n_req + n_prov
    graph = [[] for _ in range(sink + 1)]

    def add_edge(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for i in range(n_req):
        add_edge(source, req0 + i, 1, 0)
    for i, request in enumerate(requests):
        for j, provider in enumerate(providers):
            model = provider.best_for(request.requires)
            if model is None:
                continue
            # Wire constraint: an OpenAI request in JSON mode must travel
            # on a wire that can carry the mode back in its echo; the
            # Anthropic wire has no place to put it, so such a request is
            # never planned onto an Anthropic-wire provider. (For this
            # workload the JSON-mode requests are exactly the ones whose
            # requirements ask for json.)
            if request.api == "openai" and provider.wire == "anthropic" \
                    and request.requires.get("json") is True:
                continue
            price = edge_price(i, request, provider, model, requests, layout)
            if price is None:
                continue
            add_edge(req0 + i, prov0 + j, 1, price)
    for j, provider in enumerate(providers):
        add_edge(prov0 + j, sink, max(0, caps.get(provider.id, 0)), 0)

    # Successive shortest paths, one unit per request; Bellman-Ford keeps
    # the residual graph's negative reverse edges honest at this size.
    flow = 0
    while flow < n_req:
        dist = [None] * (sink + 1)
        prev = [None] * (sink + 1)
        dist[source] = 0
        for _ in range(sink + 1):
            changed = False
            for u in range(sink + 1):
                if dist[u] is None:
                    continue
                for k, edge in enumerate(graph[u]):
                    if edge[1] > 0 and (dist[edge[0]] is None
                                        or dist[u] + edge[2] < dist[edge[0]]):
                        dist[edge[0]] = dist[u] + edge[2]
                        prev[edge[0]] = (u, k)
                        changed = True
            if not changed:
                break
        if dist[sink] is None:
            break
        node = sink
        while node != source:
            u, k = prev[node]
            graph[u][k][1] -= 1
            graph[node][graph[u][k][3]][1] += 1
            node = u
        flow += 1

    plan = {}
    for i, request in enumerate(requests):
        for edge in graph[req0 + i]:
            if prov0 <= edge[0] < prov0 + n_prov and edge[1] == 0:
                provider = providers[edge[0] - prov0]
                plan[request.id] = (provider,
                                    provider.best_for(request.requires))
    return plan


def edge_price(index, request, provider, model, requests, layout):
    loss = request.weight * (request.best - model.rating) * 100
    number = None
    if layout is not None:
        position = 0
        for earlier in requests[:index]:
            if layout.get(earlier.id) == provider.id:
                position += 1
        number = provider.made + position + 1
    risk = risk_charge(provider, number)
    if risk is None:
        return None
    # Drift: an unprobed provider's slots are cheapest spent early, while
    # its first call is guaranteed ground; the probed providers' revealed
    # ground is held back for the later requests.
    drift = 0 if provider.probed else 2 * index
    return loss + risk + drift


def plan_requests(requests, providers, caps):
    """Solve, lay the plan out in request order, re-price each edge with
    the call number it would truly land on, and solve again; three rounds
    settle the numbers the risk charges are priced from."""
    layout = None
    plan = {}
    for _round in range(3):
        plan = min_cost_assignment(requests, providers, caps, layout)
        layout = {rid: choice[0].id for rid, choice in plan.items()}
    return plan


def vitality(provider):
    """How trustworthy a provider's next call is, as a walk-order group.

    Group 0 is known-good ground: the provider's last placed call answered
    (or none has been placed), and its next number is neither a burned one
    nor the dangerous unprobed second call. Group 1 is everything else
    still standing. Late in a trial the board fills with providers that
    look strong on paper and are silently spent, out, or sitting on a
    refusal slot; the failover walk reads this group first so it steps on
    the firm ground before it starts testing corpses."""
    if provider.made == 0:
        return 0
    if provider.consec > 0:
        return 1
    if provider.made >= provider.budget_low:
        # Every call from here lives on borrowed budget: the tier may
        # hold no more than its smallest size, in which case the next
        # call refuses for good. Providers that look strong and are
        # quietly spent are exactly the corpses a late walk must test
        # last, not first.
        return 1
    number = provider.next_number()
    if number in provider.burned:
        return 1
    if number == 2 and provider.probed < 2:
        return 1
    return 0


def gamble(provider):
    """For the failover walk's ordering inside a vitality group: how
    likely an unprobed second call is to be the trapped one, by the same
    density the planner prices. Other next numbers carry no comparable
    designed risk and gamble at zero."""
    if provider.next_number() == 2 and provider.probed < 2:
        return int(1500 / (provider.budget_low - 1))
    return 0


# --------------------------------------------------------------------------
# The aggregator itself: holds the providers, the plan and the serves
# already placed per provider, and answers one client request at a time.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, providers, requests, catalog):
        self.providers = providers
        self.by_id = {p.id: p for p in providers}
        self.requests = requests              # the whole manifest, in order
        self.catalog = catalog
        self.free_ids = free_id_list(catalog)
        self.served = 0                       # client requests answered
        # Serves placed per provider, failovers included: a serve spends
        # the provider's budget whichever plan it came from, so the caps
        # the planner works against are charged for every one of them.
        self.serves_done = {p.id: 0 for p in providers}
        self.plan = {}
        self.lock = threading.Lock()

    # -- planning ----------------------------------------------------------

    def remaining(self):
        return self.requests[self.served:]

    def replan(self):
        todo = self.remaining()
        if not todo:
            self.plan = {}
            return
        caps = {p.id: max(0, p.slot_cap() - self.serves_done[p.id])
                for p in self.providers}
        plan = plan_requests(todo, self.providers, caps)
        if len(plan) < len(todo):
            # The cautious caps cannot place everyone; widen them rather
            # than leave a request unplanned. Availability outranks thrift.
            wide = {p.id: max(caps[p.id], p.budget_high) for p in self.providers}
            plan = plan_requests(todo, self.providers, wide)
        self.plan = plan

    # -- serving -----------------------------------------------------------

    def candidates(self, request, parsed):
        """Every (provider, model) that could lawfully answer, best first:
        rating order, with the provider whose next call is safest breaking
        ties. Filters mirror the planner's: free, meets the requirements,
        and an OpenAI JSON-mode request never rides the Anthropic wire."""
        found = []
        for provider in self.providers:
            model = provider.best_for(request.requires)
            if model is None:
                continue
            if parsed.json_mode and provider.wire == "anthropic" \
                    and request.api == "openai":
                continue
            found.append((provider, model))
        found.sort(key=lambda choice: (vitality(choice[0]),
                                       gamble(choice[0]),
                                       -choice[1].rating))
        return found

    def serve(self, api, body):
        """Answer one client request. Returns (bytes, content type)."""
        with self.lock:
            return self._serve(api, body)

    def _serve(self, api, body):
        request = self.requests[self.served] \
            if self.served < len(self.requests) else None
        parsed = parse_openai_client(body) if api == "openai" \
            else parse_anthropic_client(body)
        if request is None:
            request = Request({"id": "extra", "api": api,
                               "stream": parsed.stream, "weight": 1,
                               "requires": {}}, 0)
        tried = set()
        target = self.plan.get(request.id)
        while True:
            choice = None
            if target is not None and target[0].id not in tried \
                    and not target[0].dead() \
                    and target[0].next_number() not in target[0].burned:
                choice = target
                target = None
            else:
                for cand in self.candidates(request, parsed):
                    if cand[0].id in tried or cand[0].dead():
                        continue
                    if cand[0].next_number() in cand[0].burned:
                        continue
                    choice = cand
                    break
                if choice is None:
                    # Nothing clean is left; take the strongest provider
                    # not yet tried, whatever its state, before giving up.
                    for cand in self.candidates(request, parsed):
                        if cand[0].id not in tried:
                            choice = cand
                            break
            if choice is None:
                payload = {"error": {"message": "no free model could serve "
                                                "the request",
                                     "type": "api_error", "param": None,
                                     "code": "no_provider"}}
                return json.dumps(payload).encode("utf-8"), "application/json"
            provider, model = choice
            tried.add(provider.id)
            answer = call_provider(provider, model, parsed)
            if answer is not None:
                self.serves_done[provider.id] += 1
                self.served += 1
                self.replan()
                return render_answer(api, parsed, model.id, answer)


# --------------------------------------------------------------------------
# The HTTP front door.
# --------------------------------------------------------------------------

def models_document(free_ids, anthropic_shape):
    if anthropic_shape:
        return {"data": [{"type": "model", "id": mid, "display_name": mid,
                          "created_at": "2026-01-01T00:00:00Z"}
                         for mid in free_ids],
                "has_more": False}
    return {"object": "list",
            "data": [{"id": mid, "object": "model", "created": CREATED,
                      "owned_by": "aggregator"} for mid in free_ids]}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    aggregator = None

    def log_message(self, *_args):
        pass

    def _reply(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.split("?")[0] != "/v1/models":
            self._reply(404, b'{"error":"not_found"}', "application/json")
            return
        agg = self.aggregator
        doc = models_document(agg.free_ids,
                              self.headers.get("x-api-key") is not None)
        self._reply(200, json.dumps(doc).encode("utf-8"), "application/json")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/v1/chat/completions", "/v1/messages"):
            self._reply(404, b'{"error":"not_found"}', "application/json")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        body = json_or_none(raw) or {}
        api = "openai" if path == "/v1/chat/completions" else "anthropic"
        payload, content_type = self.aggregator.serve(api, body)
        self._reply(200, payload, content_type)


# --------------------------------------------------------------------------
# Startup: read the brief, learn the providers' catalogs, probe the two
# richest providers, start the server, register, serve until the env says
# the workload is done, then exit 0.
# --------------------------------------------------------------------------

def wait_for_env(env_url):
    for _ in range(600):
        status, _ = http_call("GET", env_url.rstrip("/") + "/work", timeout=3)
        if status is not None:
            return
        time.sleep(0.1)


def exit_when_done(env_url, aggregator, announced):
    work_url = env_url.rstrip("/") + "/work"
    for _ in range(7200):
        status, raw = http_call("GET", work_url, timeout=3)
        doc = json_or_none(raw) if status is not None else None
        if doc is not None and doc.get("done") is True:
            time.sleep(0.2)
            os._exit(0)
        if aggregator.served >= announced and announced > 0:
            # Every announced request has been answered; the done flag
            # follows immediately, but never outwait the trial's wall.
            time.sleep(0.2)
            status, raw = http_call("GET", work_url, timeout=3)
            doc = json_or_none(raw) if status is not None else None
            if doc is not None and doc.get("done") is True:
                os._exit(0)
        time.sleep(0.05)
    os._exit(0)


def build_state(env_url):
    """Brief in, providers and catalog and manifest out."""
    brief_path = os.environ["MAKEFUN_PROMPT_FILE"]
    directory, quality, workload = load_brief(brief_path)
    providers = [Provider(entry, env_url) for entry in directory["providers"]]
    table = [Model(row) for row in quality["ratings"]]
    for provider in providers:
        provider.models = [m for m in table if m.provider == provider.id]
    best = {row["id"]: row["rating_milli"] for row in quality["best"]}
    requests = [Request(entry, best[entry["id"]])
                for entry in workload["manifest"]]
    for provider in providers:
        read_provider_catalog(provider)
    catalog = build_catalog(providers, table)
    return providers, requests, catalog


def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if "--list" in argv:
        if os.environ.get("MAKEFUN_PROMPT_FILE"):
            providers, _requests, catalog = build_state(env_url)
            sys.stdout.write(json.dumps(catalog) + "\n")
        else:
            sys.stdout.write('{"providers":[]}\n')
        return 0

    providers, requests, catalog = build_state(env_url)
    wait_for_env(env_url)

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    aggregator = Aggregator(providers, requests, catalog)
    Handler.aggregator = aggregator
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    for provider in providers:
        if provider.id in PROBE_PROVIDERS:
            for _ in range(PROBE_CALLS):
                place_probe(provider)

    aggregator.replan()

    status, raw = http_call(
        "POST", env_url.rstrip("/") + "/register",
        headers={"Content-Type": "application/json"},
        body=json.dumps({"protocol_version": 1, "port": port,
                         "catalog": catalog}),
        timeout=10)
    announced = len(requests)
    doc = json_or_none(raw) if status is not None else None
    if doc is not None and isinstance(doc.get("requests"), int):
        announced = doc["requests"]

    watcher = threading.Thread(target=exit_when_done,
                               args=(env_url, aggregator, announced),
                               daemon=True)
    watcher.start()
    thread.join()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
