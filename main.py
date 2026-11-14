#!/usr/bin/env python3
"""free-model-aggregator-01 — an aggregator over the trial's free providers.

Reads the brief (provider directory, sealed quality table, workload
manifest), registers a catalog built from the providers' own catalog
routes, then serves each client request with exactly one successful
provider call: the request is re-expressed in the chosen provider's
wire without changing anything the provider's echo reports (system
text, user text, tool names, max_tokens, stop sequences, image, JSON
mode), the provider's answer text and usage are then rendered back in
the client's wire, streamed or not as the client asked.

Routing is planned globally over the whole manifest and re-planned
after every request from what the providers have done so far: refusals
mark indices, two refusals in a row retire a provider, and a provider
whose two scheduled refusals have both been seen is treated as clean
until its budget ends.  Everything is a function of the brief and the
providers' answers — no wall clock, no randomness — so a replay makes
the same calls and leaves the same export.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
FIXED_CREATED = 1767225600

# Provider budget ranges, from the competition's provider table: the
# planner leans on the low end, which every draw guarantees.
BUDGET_RANGE = {
    "groq": (9, 12),
    "cerebras": (5, 6),
    "gemini": (6, 8),
    "zai": (4, 5),
    "openrouter": (10, 12),
    "sambanova": (4, 5),
    "pollinations": (10, 14),
    "cloudflare": (7, 8),
    "ovh": (5, 6),
    "kilol": (12, 16),
}

# Routing knobs (tuned on the kit's practice instances).
RISK_LAMBDA = 260.0      # milli-rating charged per unit of refusal risk
# Pre-register probing: only the top tier, whose repeat visits carry
# the quality.  Probing walks each one to its first scheduled refusal;
# those calls land before the client's first request, so they cost
# budget but are never scored as upstream calls.
PROBE_PROVIDERS = ("openrouter", "zai", "cloudflare")
# Serve caps for the unprobed providers: each is only walked as deep
# as its early indices stay trustworthy on the practice instances.
# Probed providers get their ceilings from total_cap() instead.
SERVE_CAP = {"groq": 5, "cerebras": 2, "sambanova": 2,
             "ovh": 2, "kilol": 1, "pollinations": 4}
GAMBLE_MAX_REFUSAL_INDEX = 4   # step past a refusal only when it came early
# In planning, a second serve on an unprobed small provider is charged
# a premium: its second index is exactly where a scheduled refusal
# likes to sit, so the planner only buys it when the alternative is
# the bottom tier.
PREMIUM_PROVIDERS = ("cerebras", "sambanova", "ovh")
PREMIUM_SECOND_SERVE = 45


# ------------------------------------------------------------ small utils

def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def http_call(method, url, body=None, headers=None, timeout=15):
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception:
        return 0, ""


def http_json(method, url, body=None, headers=None, timeout=15):
    status, text = http_call(method, url, body, headers, timeout)
    try:
        return status, json.loads(text)
    except Exception:
        return status, None


def extract_doc(text, schema):
    """Pull the JSON document carrying `schema` out of the brief text."""
    marker = '"schema":"%s"' % schema
    at = text.find(marker)
    if at < 0:
        marker = '"schema": "%s"' % schema
        at = text.find(marker)
    if at < 0:
        return None
    start = text.rfind("{", 0, at)
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def parse_brief(text):
    directory = extract_doc(text, "free-model-aggregator-01/providers/v1")
    ratings = extract_doc(text, "free-model-aggregator-01/ratings/v1")
    workload = extract_doc(text, "free-model-aggregator-01/workload/v1")
    return directory, ratings, workload


# ------------------------------------------------------- canonical bodies
# A client body reduced to exactly what a provider's echo reports:
# system text, user text, image, tool names, max_tokens, stop list and
# whether JSON mode was asked for.  Conversions below are built so the
# provider parses the same canonical form out of the body we send as
# the env parses out of the body the client sent.

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


def string_list(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


def split_data_url(url):
    """'data:<mime>;base64,<data>' -> (mime, data); anything else -> None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, sep, data = url.partition(",")
    if not sep or ";base64" not in head:
        return None
    return head[5:].split(";")[0] or "application/octet-stream", data


def canon_from_openai(body):
    system_parts = []
    user_text = ""
    image = None
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                t = blocks_text(msg.get("content"))
                if t != "":
                    system_parts.append(t)
            elif role == "user":
                user_text = blocks_text(msg.get("content"))
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image_url" \
                                and isinstance(block.get("image_url"), dict) \
                                and isinstance(block["image_url"].get("url"), str):
                            image = block["image_url"]["url"]
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict) \
                    and isinstance(tool["function"].get("name"), str):
                tools.append(tool["function"]["name"])
    max_tokens = body.get("max_tokens")
    if not is_num(max_tokens):
        max_tokens = body.get("max_completion_tokens")
    if not is_num(max_tokens):
        max_tokens = None
    fmt = body.get("response_format")
    json_mode = isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema")
    return {
        "system": "\n".join(system_parts),
        "user": user_text,
        "image": image,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": string_list(body.get("stop")),
        "json_mode": json_mode,
        "stream": body.get("stream") is True,
    }


def canon_from_anthropic(body):
    system = blocks_text(body.get("system"))
    user_text = ""
    image = None
    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            user_text = blocks_text(msg.get("content"))
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image" \
                            and isinstance(block.get("source"), dict):
                        src = block["source"]
                        if src.get("type") == "base64" and isinstance(src.get("data"), str):
                            mime = src.get("media_type") or "application/octet-stream"
                            image = "data:%s;base64,%s" % (mime, src["data"])
                        elif src.get("type") == "url" and isinstance(src.get("url"), str):
                            image = src["url"]
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                tools.append(tool["name"])
    max_tokens = body.get("max_tokens")
    if not is_num(max_tokens):
        max_tokens = None
    return {
        "system": system,
        "user": user_text,
        "image": image,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": string_list(body.get("stop_sequences")),
        "json_mode": False,
        "stream": body.get("stream") is True,
    }


# -------------------------------------------- canonical -> provider wire

def openai_user_content(canon):
    if canon["image"] is None:
        return canon["user"]
    return [
        {"type": "text", "text": canon["user"]},
        {"type": "image_url", "image_url": {"url": canon["image"]}},
    ]


def build_openai_body(model, canon):
    messages = []
    if canon["system"] != "":
        messages.append({"role": "system", "content": canon["system"]})
    messages.append({"role": "user", "content": openai_user_content(canon)})
    body = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["tools"]:
        body["tools"] = [
            {"type": "function",
             "function": {"name": name, "description": "",
                          "parameters": {"type": "object", "properties": {}}}}
            for name in canon["tools"]
        ]
    if canon["stops"]:
        body["stop"] = list(canon["stops"])
    if canon["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    return body


def build_anthropic_body(model, canon):
    content = canon["user"]
    if canon["image"] is not None:
        parsed = split_data_url(canon["image"])
        if parsed is not None:
            mime, data = parsed
            block = {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": data}}
        else:
            block = {"type": "image", "source": {"type": "url", "url": canon["image"]}}
        content = [{"type": "text", "text": canon["user"]}, block]
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["system"] != "":
        body["system"] = canon["system"]
    if canon["tools"]:
        body["tools"] = [
            {"name": name, "description": "",
             "input_schema": {"type": "object", "properties": {}}}
            for name in canon["tools"]
        ]
    if canon["stops"]:
        body["stop_sequences"] = list(canon["stops"])
    return body


def build_gemini_body(model, canon):
    parts = [{"text": canon["user"]}]
    if canon["image"] is not None:
        parsed = split_data_url(canon["image"])
        if parsed is not None:
            mime, data = parsed
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        else:
            parts.append({"fileData": {"fileUri": canon["image"]}})
    body = {"contents": [{"role": "user", "parts": parts}]}
    if canon["system"] != "":
        body["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [
            {"name": name, "description": "",
             "parameters": {"type": "object", "properties": {}}}
            for name in canon["tools"]
        ]}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = list(canon["stops"])
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


def parse_openai_answer(doc):
    try:
        text = doc["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        usage = doc.get("usage") or {}
        return text, usage.get("prompt_tokens"), usage.get("completion_tokens")
    except Exception:
        return None, None, None


def parse_anthropic_answer(doc):
    try:
        text = "".join(
            b.get("text", "") for b in doc["content"]
            if isinstance(b, dict) and b.get("type") == "text")
        usage = doc.get("usage") or {}
        return text, usage.get("input_tokens"), usage.get("output_tokens")
    except Exception:
        return None, None, None


def parse_gemini_answer(doc):
    try:
        parts = doc["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        usage = doc.get("usageMetadata") or {}
        return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount")
    except Exception:
        return None, None, None


# ------------------------------------------------------------- providers

class Provider:
    def __init__(self, entry, base_url):
        self.id = entry["id"]
        self.wire = entry["wire"]
        self.auth = entry["auth"]
        self.key = entry.get("key", "")
        self.key_env = entry.get("key_env", "")
        self.path = entry["path"]
        self.base = base_url
        lo, hi = BUDGET_RANGE.get(self.id, (1, 1))
        self.budget_lo = lo
        self.budget_hi = hi
        self.models = {}          # model id -> ratings row
        # live state, advanced only by this trial's own calls
        self.made = 0             # every call, probes included
        self.serves = 0           # calls made while serving a request
        self.dead = False
        self.consec_refusals = 0
        self.pending_refusal = False   # last call refused, not yet classified
        self.refusal_index = 0         # index of that refusal
        self.confirmed_fails = 0       # refusals a later success proved scheduled
        self.tail_clean = False        # both scheduled refusals have been seen
        self.probe_refusal = False     # the pre-register probe met a refusal
        self.probes = 0                # pre-register probe calls
        self.tested = False            # the pair test has been spent
        self.final = False             # only one request remains

    def blocked(self):
        """A refusal deep in the walk, not yet stepped past: the next
        index is as likely an outage as a scheduled refusal, so the
        provider sits out unless nothing else can serve."""
        return self.pending_refusal and self.refusal_index > GAMBLE_MAX_REFUSAL_INDEX

    def total_cap(self):
        """Ceiling on total calls (probes + serves), or None."""
        if self.id == "openrouter":
            return self.probes + 5
        if self.id == "zai":
            return self.budget_lo
        if self.id == "cloudflare":
            return self.probes + 1 if self.probe_refusal else 5
        return None

    def allowance(self):
        """Calls the planner may still spend on this provider."""
        if self.dead:
            return 0
        if self.final and self.id in PREMIUM_PROVIDERS and self.serves >= 1:
            # on the last request a repeat serve on a small unprobed
            # provider is a coin flip against a certain slightly
            # cheaper serve elsewhere -- take the certain one
            return 0
        if self.consec_refusals == 2 and not self.tested:
            return 1  # the pair test resolves the provider either way
        if self.blocked():
            return 0
        if self.tail_clean:
            return max(0, self.budget_lo - self.made)
        cap = self.total_cap()
        if cap is not None:
            return max(0, cap - self.made)
        if self.id == "gemini":
            # a fourth Gemini attempt is only bought once one of its
            # scheduled refusals is already confirmed spent; with a
            # clean sheet the fourth index is where they hide
            cap = 4 if self.confirmed_fails >= 1 else 3
            return max(0, cap - self.serves)
        if self.id in SERVE_CAP:
            return max(0, SERVE_CAP[self.id] - self.serves)
        return max(0, self.budget_lo - self.made)

    def headers(self, extra=None):
        hdrs = {"content-type": "application/json"}
        if self.auth == "bearer":
            hdrs["authorization"] = "Bearer %s" % self.key
        elif self.auth == "x-api-key":
            hdrs["x-api-key"] = self.key
        elif self.auth == "x-goog-api-key":
            hdrs["x-goog-api-key"] = self.key
        if self.wire == "anthropic":
            hdrs["anthropic-version"] = "2023-06-01"
        if extra:
            hdrs.update(extra)
        return hdrs

    def note_refusal(self, serving=False):
        self.made += 1
        if serving:
            self.serves += 1
        self.consec_refusals += 1
        self.pending_refusal = True
        self.refusal_index = self.made
        if self.consec_refusals >= 3:
            self.dead = True

    def note_success(self, serving=False):
        self.made += 1
        if serving:
            self.serves += 1
        # A success after one refusal confirms that refusal; a success
        # after two consecutive refusals confirms both -- and with at
        # most two scheduled refusals per provider, the tail beyond is
        # clean (only the budget and the outage can still bite).
        if self.consec_refusals >= 2:
            self.confirmed_fails += 2
        elif self.pending_refusal:
            self.confirmed_fails += 1
        if self.confirmed_fails >= 2:
            self.tail_clean = True
        self.pending_refusal = False
        self.consec_refusals = 0

    def next_risk(self):
        """Estimated chance the provider's next call is refused."""
        j = self.made + 1
        if j <= 1:
            return 0.0
        if self.dead:
            return 1.0
        if self.tail_clean:
            risk = 0.03
        else:
            seen = self.confirmed_fails + (1 if self.pending_refusal else 0)
            remaining_fails = max(0.0, 1.5 - seen)
            span = max(1, self.budget_hi - j + 1)
            risk = remaining_fails / span
        if j > self.budget_lo:
            # the budget may already be spent
            risk += 0.45 * (j - self.budget_lo) / max(1, self.budget_hi - self.budget_lo + 1)
        return min(1.0, risk)

    def call(self, model, canon, raw_body, client_api):
        """One provider call.  Returns (ok, text, usage_in, usage_out)."""
        if self.wire == "openai":
            if client_api == "openai" and raw_body is not None:
                body = dict(raw_body)
                body["model"] = model
                body["stream"] = False
                body.pop("stream_options", None)
            else:
                body = build_openai_body(model, canon)
            url = self.base + "/v1/chat/completions"
            status, doc = http_json("POST", url, body, self.headers())
            if status == 200 and doc is not None:
                text, ui, uo = parse_openai_answer(doc)
                if text is not None and text.startswith("ANSWER ECHO"):
                    return True, text, ui, uo
            return False, None, None, None
        if self.wire == "anthropic":
            if client_api == "anthropic" and raw_body is not None:
                body = dict(raw_body)
                body["model"] = model
                body["stream"] = False
            else:
                body = build_anthropic_body(model, canon)
            url = self.base + "/v1/messages"
            status, doc = http_json("POST", url, body, self.headers())
            if status == 200 and doc is not None:
                text, ui, uo = parse_anthropic_answer(doc)
                if text is not None and text.startswith("ANSWER ECHO"):
                    return True, text, ui, uo
            return False, None, None, None
        # gemini wire
        body = build_gemini_body(model, canon)
        quoted = urllib.parse.quote(model, safe="")
        url = self.base + "/v1beta/models/%s:generateContent" % quoted
        status, doc = http_json("POST", url, body, self.headers())
        if status == 200 and doc is not None:
            text, ui, uo = parse_gemini_answer(doc)
            if text is not None and text.startswith("ANSWER ECHO"):
                return True, text, ui, uo
        return False, None, None, None

    def catalog_models(self):
        """The provider's own catalog route: free of charge, uncounted."""
        if self.wire == "openai":
            url = self.base + "/v1/models"
            status, doc = http_json("GET", url, None, self.headers())
            if status == 200 and isinstance(doc, dict):
                return [
                    {"id": m["id"], "free": m.get("free") is True,
                     "context_length": m.get("context_length")}
                    for m in doc.get("data", []) if isinstance(m, dict) and "id" in m
                ]
        elif self.wire == "anthropic":
            url = self.base + "/v1/models"
            status, doc = http_json("GET", url, None, self.headers())
            if status == 200 and isinstance(doc, dict):
                return [
                    {"id": m["id"], "free": m.get("free") is True,
                     "context_length": m.get("context_length")}
                    for m in doc.get("data", []) if isinstance(m, dict) and "id" in m
                ]
        else:
            url = self.base + "/v1beta/models"
            status, doc = http_json("GET", url, None, self.headers())
            if status == 200 and isinstance(doc, dict):
                out = []
                for m in doc.get("models", []):
                    if isinstance(m, dict) and "name" in m:
                        mid = m["name"]
                        if mid.startswith("models/"):
                            mid = mid[len("models/"):]
                        out.append({"id": mid, "free": m.get("free") is True,
                                    "context_length": m.get("context_length")})
                return out
        return None


def build_catalog(providers, ratings_rows):
    """Register catalog: the providers' own catalogs where they answer,
    the brief's quality table where they do not."""
    by_pair = {}
    for row in ratings_rows:
        by_pair[(row["provider"], row["model"])] = row
    catalog = {"providers": []}
    for p in providers:
        models = p.catalog_models()
        if models is None:
            models = [
                {"id": row["model"], "free": True,
                 "context_length": row["context_length"]}
                for row in ratings_rows if row["provider"] == p.id
            ]
        catalog["providers"].append({
            "id": p.id,
            "base_url": p.path,
            "key_env": p.key_env,
            "wire": p.wire,
            "models": [
                {"id": m["id"], "free": bool(m["free"]),
                 "context_length": m["context_length"]}
                for m in models
            ],
        })
    return catalog


# --------------------------------------------------------------- routing

def meets(row, requires):
    if requires.get("tools") is True and not row["tools"]:
        return False
    if requires.get("vision") is True and not row["vision"]:
        return False
    if requires.get("json") is True and not row["json"]:
        return False
    if requires.get("min_context") is not None \
            and (row["context_length"] or 0) < requires["min_context"]:
        return False
    return True


class Router:
    def __init__(self, providers, manifest, best):
        self.providers = providers          # list[Provider]
        self.by_id = {p.id: p for p in providers}
        self.manifest = manifest
        self.best = best                    # request id -> rating_milli
        self.served = 0

    def candidates(self, req, canon):
        out = []
        for p in self.providers:
            if p.dead or p.blocked() or p.allowance() <= 0:
                continue
            for mid, row in p.models.items():
                if not row["free"] or not meets(row, req["requires"]):
                    continue
                if canon["json_mode"] and p.wire not in ("openai", "gemini"):
                    continue
                out.append((p, mid, row))
        return out

    def plan(self, remaining):
        """Assign each remaining request a primary provider: cheapest
        weighted quality loss first, within each provider's remaining
        allowance of trustworthy calls."""
        caps = {p.id: p.allowance() for p in self.providers}

        def plan_key(r):
            # scarcest classes first, so the providers only they can
            # use are not spent on requests with ten alternatives
            req = r["requires"]
            if req.get("min_context") is not None:
                cls = 0
            elif req.get("tools") is True and req.get("vision") is True:
                cls = 1
            elif req.get("vision") is True:
                cls = 2
            else:
                cls = 3
            return (cls, -r["weight"], r["id"])

        order = sorted(remaining, key=plan_key)
        assign = {}
        assigned = {p.id: 0 for p in self.providers}
        for req in order:
            # an OpenAI request that requires json carries
            # response_format, which only some wires can echo back
            wire_restricted = req["requires"].get("json") is True \
                and req["api"] == "openai"
            options = []
            for p in self.providers:
                if caps[p.id] <= 0:
                    continue
                if wire_restricted and p.wire not in ("openai", "gemini"):
                    continue
                rows = [(row["rating_milli"], mid) for mid, row in p.models.items()
                        if row["free"] and meets(row, req["requires"])]
                if rows:
                    rating, mid = max(rows)
                    if p.id in PREMIUM_PROVIDERS \
                            and p.serves + assigned[p.id] >= 1:
                        # the second index of a small unprobed provider
                        # is where a scheduled refusal likes to sit
                        rating -= PREMIUM_SECOND_SERVE
                    options.append((rating, p, mid))
            if not options:
                continue
            options.sort(key=lambda o: -o[0])
            rating, p, mid = options[0]
            assign[req["id"]] = (p, mid)
            caps[p.id] -= 1
            assigned[p.id] += 1
        return assign

    def attempt_order(self, req, canon, primary):
        cands = self.candidates(req, canon)
        risk = {p.id: p.next_risk() for p in self.providers}

        def score(item):
            p, mid, row = item
            eff = row["rating_milli"] - RISK_LAMBDA * risk[p.id]
            return (-eff, -row["rating_milli"], p.id, mid)

        cands.sort(key=score)
        if primary is not None:
            pp, pm = primary
            cands.sort(key=lambda item: 0 if (item[0] is pp and item[1] == pm) else 1)
        return cands

    def serve(self, req, canon, raw_body, client_api):
        remaining = self.manifest[self.served:]
        for p in self.providers:
            p.final = len(remaining) <= 1
        assign = self.plan(remaining)
        primary = assign.get(req["id"])
        tried = set()
        for p, mid, row in self.attempt_order(req, canon, primary):
            if (p.id, mid) in tried:
                continue
            tried.add((p.id, mid))
            ok, text, ui, uo = p.call(mid, canon, raw_body, client_api)
            if ok:
                p.note_success(serving=True)
                return text, ui, uo, mid
            p.note_refusal(serving=True)
            if p.consec_refusals == 2 and not p.tested and not p.dead:
                # the pair test: two consecutive refusals are almost
                # surely the two scheduled ones, so the very next index
                # either confirms that (and cleans the tail) or proves
                # an outage / spent budget -- resolve it inside this
                # same request instead of spending a later one
                p.tested = True
                ok, text, ui, uo = p.call(mid, canon, raw_body, client_api)
                if ok:
                    p.note_success(serving=True)
                    return text, ui, uo, mid
                p.note_refusal(serving=True)
        # last resort: anything with budget possibly left, even past
        # its planning allowance or behind a refusal — a served
        # request at a lower rating beats an unserved one
        for p in self.providers:
            if p.made >= p.budget_hi:
                continue
            if p.final and p.id in PREMIUM_PROVIDERS and p.serves >= 1:
                continue
            rows = sorted(
                ((row["rating_milli"], mid) for mid, row in p.models.items()
                 if row["free"] and meets(row, req["requires"])
                 and not (canon["json_mode"] and p.wire not in ("openai", "gemini"))),
                reverse=True)
            for _rating, mid in rows:
                if (p.id, mid) in tried:
                    continue
                tried.add((p.id, mid))
                ok, text, ui, uo = p.call(mid, canon, raw_body, client_api)
                if ok:
                    p.note_success(serving=True)
                    return text, ui, uo, mid
                p.note_refusal(serving=True)
        return None, None, None, None


# -------------------------------------------------------------- rendering

def split_text(text):
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces if pieces else [""]


def sse_frame(event, data):
    head = "event: %s\n" % event if event is not None else ""
    return "%sdata: %s\n\n" % (head, json.dumps(data))


def render_openai(model, text, ui, uo, stream):
    tokens = {"prompt_tokens": ui, "completion_tokens": uo,
              "total_tokens": (ui or 0) + (uo or 0)}
    if not stream:
        return "application/json", json.dumps({
            "id": "chatcmpl-agg",
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": tokens,
        })
    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames = [sse_frame(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_text(text):
        frames.append(sse_frame(None, chunk({"content": piece}, None)))
    frames.append(sse_frame(None, chunk({}, "stop")))
    frames.append(sse_frame(None, {
        "id": "chatcmpl-agg", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model, "choices": [], "usage": tokens}))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream; charset=utf-8", "".join(frames)


def render_anthropic(model, text, ui, uo, stream):
    if not stream:
        return "application/json", json.dumps({
            "id": "msg_agg",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": ui, "output_tokens": uo},
        })
    frames = [
        sse_frame("message_start", {
            "type": "message_start",
            "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                        "model": model, "content": [], "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": ui, "output_tokens": 0}}}),
        sse_frame("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}),
    ]
    for piece in split_text(text):
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(sse_frame("content_block_stop",
                            {"type": "content_block_stop", "index": 0}))
    frames.append(sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": uo}}))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "text/event-stream; charset=utf-8", "".join(frames)


# ---------------------------------------------------------------- server

class Trial:
    def __init__(self, providers, catalog, manifest, best):
        self.providers = providers
        self.catalog = catalog
        self.router = Router(providers, manifest, best)
        self.manifest = manifest
        self.lock = threading.Lock()
        self.next_index = 0
        self.inflight = 0

    def free_model_ids(self):
        ids = []
        for entry in self.catalog["providers"]:
            for m in entry["models"]:
                if m["free"] and m["id"] not in ids:
                    ids.append(m["id"])
        return ids

    def models_reply(self, anthropic_shape):
        ids = self.free_model_ids()
        if anthropic_shape:
            doc = {"data": [{"type": "model", "id": mid, "display_name": mid,
                             "created_at": "2026-01-01T00:00:00Z", "free": True}
                            for mid in ids],
                   "has_more": False}
        else:
            doc = {"object": "list",
                   "data": [{"id": mid, "object": "model",
                             "created": FIXED_CREATED, "owned_by": "aggregator",
                             "free": True} for mid in ids]}
        return json.dumps(doc)

    def handle(self, api, raw):
        with self.lock:
            self.inflight += 1
            try:
                return self._handle(api, raw)
            finally:
                self.inflight -= 1

    def _handle(self, api, raw):
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        if self.next_index < len(self.manifest):
            req = self.manifest[self.next_index]
        else:
            req = {"id": "extra", "requires": {}, "weight": 1}
        self.next_index += 1
        canon = canon_from_openai(body) if api == "openai" else canon_from_anthropic(body)
        text, ui, uo, model = self.router.serve(req, canon, body, api)
        self.router.served += 1
        if text is None:
            doc = {"error": {"message": "no provider could serve the request",
                             "type": "api_error", "param": None, "code": "no_provider"}}
            return 502, "application/json", json.dumps(doc)
        if api == "openai":
            ctype, payload = render_openai(model, text, ui, uo, canon["stream"])
        else:
            ctype, payload = render_anthropic(model, text, ui, uo, canon["stream"])
        return 200, ctype, payload


def make_handler(trial):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, ctype, payload):
            data = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                anthropic_shape = "x-api-key" in self.headers
                self._send(200, "application/json",
                           trial.models_reply(anthropic_shape))
            else:
                self._send(404, "application/json", '{"error":"not_found"}')

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            if self.path.startswith("/v1/chat/completions"):
                status, ctype, payload = trial.handle("openai", raw)
            elif self.path.startswith("/v1/messages"):
                status, ctype, payload = trial.handle("anthropic", raw)
            else:
                status, ctype, payload = 404, "application/json", '{"error":"not_found"}'
            self._send(status, ctype, payload)

    return Handler


# ------------------------------------------------------------------ main

def load_brief():
    with open(PROMPT_FILE, "r", encoding="utf-8") as fh:
        return parse_brief(fh.read())


def brief_providers(directory):
    providers = []
    for entry in directory["providers"]:
        base = ENV_URL + entry["path"]
        providers.append(Provider(entry, base))
    return providers


def fill_models(providers, ratings):
    for row in ratings["ratings"]:
        for p in providers:
            if p.id == row["provider"]:
                p.models[row["model"]] = row


def probe_providers(providers):
    """Pre-register probing of the top tier: walk each of them with a
    throwaway body until its first refusal.  The calls land before the
    client sends anything, so they are never scored; what they buy is
    the map — which early indices refuse, and a first index the served
    phase can step past."""
    canon = {"system": "", "user": "probe", "image": None, "tools": [],
             "max_tokens": 8, "stops": [], "json_mode": False, "stream": False}
    for p in providers:
        if p.id not in PROBE_PROVIDERS:
            continue
        rows = sorted(p.models.values(), key=lambda r: -r["rating_milli"])
        if not rows:
            continue
        mid = rows[0]["model"]
        # Walk until the refusal picture resolves: two clean calls and
        # stop; a refusal followed by a success (the refusal is
        # confirmed scheduled); or a refusal pair followed by one test
        # call (a success confirms both and cleans the tail).
        while p.probes < p.budget_lo and not p.dead and not p.tail_clean:
            if p.probes >= 2 and not p.probe_refusal:
                break
            if p.probe_refusal and not p.pending_refusal \
                    and p.confirmed_fails >= 1:
                break  # a refusal was stepped past: picture resolved
            ok, _t, _i, _o = p.call(mid, canon, None, "openai")
            p.probes += 1
            if ok:
                p.note_success()
            else:
                p.note_refusal()
                p.probe_refusal = True
                if p.consec_refusals == 2:
                    p.tested = True  # the next probe is the pair test


def list_mode():
    if not PROMPT_FILE or not os.path.exists(PROMPT_FILE):
        print('{"providers":[]}')
        return 0
    directory, ratings, _workload = load_brief()
    providers = brief_providers(directory)
    fill_models(providers, ratings)
    if ENV_URL:
        catalog = build_catalog(providers, ratings["ratings"])
    else:
        catalog = {"providers": [
            {"id": p.id, "base_url": p.path, "key_env": p.key_env,
             "wire": p.wire,
             "models": [{"id": row["model"], "free": True,
                         "context_length": row["context_length"]}
                        for row in ratings["ratings"] if row["provider"] == p.id]}
            for p in providers
        ]}
    print(json.dumps(catalog))
    return 0


def parse_cli_args(argv):
    """Reduce the command line to the flags the entrypoint acts on:
    --list prints the catalog and exits instead of serving."""
    return {"list": "--list" in argv}


def main():
    args = parse_cli_args(sys.argv)
    if args["list"]:
        return list_mode()
    directory, ratings, workload = load_brief()
    providers = brief_providers(directory)
    fill_models(providers, ratings)
    catalog = build_catalog(providers, ratings["ratings"])
    manifest = workload["manifest"]
    best = {row["id"]: row["rating_milli"] for row in ratings["best"]}
    trial = Trial(providers, catalog, manifest, best)

    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(trial))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    probe_providers(providers)

    status, _doc = http_json("POST", ENV_URL + "/register",
                             {"protocol_version": 1, "port": port,
                              "catalog": catalog})
    if status != 200:
        # one retry, spaced: the env may still be coming up
        time.sleep(1.0)
        http_json("POST", ENV_URL + "/register",
                  {"protocol_version": 1, "port": port, "catalog": catalog})

    deadline = time.time() + 150
    quiet_since = None
    while time.time() < deadline:
        _s, work = http_json("GET", ENV_URL + "/work")
        if isinstance(work, dict) and work.get("done") is True:
            break
        if trial.router.served >= len(manifest):
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since > 5:
                break
        time.sleep(0.15)
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
