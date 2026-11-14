#!/usr/bin/env python3
"""Free-model aggregator: one server, two wires, ten free providers.

The server reads its brief (MAKEFUN_PROMPT_FILE), asks every provider for
its own model list, registers the combined catalog with the trial env, and
then answers the env client request by request. Each request is served by
the highest-rated free model that can meet it; when a provider refuses, the
next best candidate takes over. The server speaks the OpenAI Chat
Completions wire and the Anthropic Messages wire to the client, in JSON and
in server-sent events, and translates each request into whichever of the
three provider wires (OpenAI, Anthropic, Gemini) the chosen provider uses,
keeping the request's system prompt, user text, images, tools, stop
sequences, max_tokens and JSON mode intact so the provider echoes them back
exactly as the client sent them.

Every choice is a function of the brief, the requests and the providers'
answers: no wall clock, no unseeded randomness, so a replay leaves the same
state export. Standard library only.
"""

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
CREATED = 1767225600
ANTHROPIC_VERSION = "2023-06-01"

# Stop-reason families, shared by all three wires: 0 ended normally,
# 1 hit the length cap, 2 stopped to call a tool, 3 was refused.
STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOPS = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOPS = ["end_turn", "max_tokens", "tool_use", "refusal"]
WIRE_STOPS = {"openai": "stop", "anthropic": "end_turn", "gemini": "STOP"}


def stop_in_wire(stop, wire):
    """Map a provider stop reason onto the client's wire, same family."""
    family = STOP_FAMILY.get(stop, 0)
    if wire == "anthropic":
        return ANTHROPIC_STOPS[family]
    return OPENAI_STOPS[family]


def strip_none(mapping):
    return {key: value for key, value in mapping.items() if value is not None}


def to_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def http_call(method, url, headers, body=None, timeout=10):
    """One HTTP call; returns (status, raw bytes). 0 means unreachable."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        try:
            return error.code, error.read()
        except Exception:
            return error.code, b""
    except Exception:
        return 0, b""


def extract_docs(text):
    """Pull the JSON documents embedded in the brief, keyed by schema."""
    docs = {}
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        at = text.find('{"schema"', cursor)
        if at < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[at:])
        except json.JSONDecodeError:
            cursor = at + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("schema"), str):
            docs[obj["schema"]] = obj
        cursor = at + end
    return docs


def load_brief(path):
    """Read the brief into providers, ratings and the workload manifest."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    docs = extract_docs(text)
    providers = docs.get("free-model-aggregator-01/providers/v1", {}).get("providers")
    ratings = docs.get("free-model-aggregator-01/ratings/v1", {}).get("ratings")
    manifest = docs.get("free-model-aggregator-01/workload/v1", {}).get("manifest")
    if not isinstance(providers, list) or not isinstance(ratings, list):
        return None
    manifest_map = {}
    if isinstance(manifest, list):
        for item in manifest:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                manifest_map[item["id"]] = item
    return {"providers": providers, "ratings": ratings, "manifest": manifest_map}


def auth_headers(provider):
    """The header that carries one provider's key, by its own scheme."""
    scheme = provider.get("auth")
    key = provider.get("key") or ""
    if scheme == "bearer":
        return {"Authorization": "Bearer " + key}
    if scheme == "x-api-key":
        return {"x-api-key": key}
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def probe_catalog(env_url, provider):
    """Ask one provider for its model list; None when it will not answer."""
    base = env_url.rstrip("/") + provider.get("path", "")
    headers = auth_headers(provider)
    wire = provider.get("wire")
    if wire == "gemini":
        status, raw = http_call("GET", base + "/v1beta/models", headers, timeout=10)
        if status != 200:
            return None
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        found = []
        for item in doc.get("models") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or ""
            model_id = name[7:] if name.startswith("models/") else name
            if not model_id:
                continue
            context = item.get("context_length", item.get("inputTokenLimit"))
            found.append((model_id, item.get("free") is True, context))
        return found
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, raw = http_call("GET", base + "/v1/models", headers, timeout=10)
    if status != 200:
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    found = []
    for item in doc.get("data") or []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        found.append((model_id, item.get("free") is True, item.get("context_length")))
    return found


def build_catalog(env_url, providers, ratings):
    """The catalog document: every provider, its path, wire and models."""
    by_key = {}
    for row in ratings:
        by_key[(row.get("provider"), row.get("model"))] = row
    entries = []
    for provider in providers:
        models = probe_catalog(env_url, provider)
        if models is None:
            # The provider would not list its models: fall back to the
            # brief's quality table, which names its free models.
            models = [
                (row.get("model"), True, row.get("context_length"))
                for row in ratings
                if row.get("provider") == provider.get("id")
            ]
        fixed = []
        for model_id, free, context in models:
            if context is None:
                row = by_key.get((provider.get("id"), model_id))
                context = row.get("context_length") if row else None
            fixed.append(
                {"id": model_id, "free": bool(free), "context_length": context}
            )
        entries.append(
            {
                "id": provider.get("id"),
                "base_url": provider.get("path"),
                "key_env": provider.get("key_env"),
                "wire": provider.get("wire"),
                "models": fixed,
            }
        )
    return {"providers": entries}


# ---------------------------------------------------------------------------
# Request normalisation: both client wires into one neutral shape.
#
# A block is {"text": ...}, {"image_b64": (mime, data)} or {"image_url": ...}.
# The neutral shape keeps everything the provider echoes back: system text,
# user text, images, tool names, max_tokens, stop sequences, JSON mode.
# ---------------------------------------------------------------------------


def text_of_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def image_block_from_url(url):
    if isinstance(url, str) and url.startswith("data:"):
        head, _, data = url.partition(",")
        mime = head[5:].split(";")[0] or "image/png"
        return {"image_b64": (mime, data)}
    return {"image_url": url}


def blocks_of_content(content, api):
    if isinstance(content, str):
        return [{"text": content}] if content else []
    blocks = []
    if not isinstance(content, list):
        return blocks
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            blocks.append({"text": item.get("text", "")})
        elif api == "openai" and kind == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
            blocks.append(image_block_from_url(url))
        elif api == "anthropic" and kind == "image":
            source = item.get("source") or {}
            if source.get("type") == "base64":
                blocks.append(
                    {
                        "image_b64": (
                            source.get("media_type", "image/png"),
                            source.get("data", ""),
                        )
                    }
                )
            elif source.get("type") == "url":
                blocks.append({"image_url": source.get("url", "")})
    return blocks


def normalize_request(body, api):
    """Reduce one client body to the neutral shape both wires share."""
    system_text = ""
    messages = []
    tools = []
    if api == "openai":
        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            blocks = blocks_of_content(message.get("content"), "openai")
            if role == "system":
                text = "\n".join(b["text"] for b in blocks if "text" in b)
                system_text = (system_text + "\n" + text) if system_text else text
            else:
                messages.append((role, blocks))
        for tool in body.get("tools") or []:
            fn = (tool or {}).get("function") or {}
            if fn.get("name"):
                tools.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description"),
                        "parameters": fn.get("parameters"),
                    }
                )
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_completion_tokens")
        stops = to_list(body.get("stop"))
        fmt = body.get("response_format")
        json_mode = isinstance(fmt, dict) and fmt.get("type") in (
            "json_object",
            "json_schema",
        )
        tool_choice = body.get("tool_choice")
    else:
        system_text = text_of_content(body.get("system"))
        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            messages.append(
                (message.get("role"), blocks_of_content(message.get("content"), "anthropic"))
            )
        for tool in body.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name"):
                tools.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description"),
                        "parameters": tool.get("input_schema"),
                    }
                )
        max_tokens = body.get("max_tokens")
        stops = to_list(body.get("stop_sequences"))
        json_mode = False
        tool_choice = body.get("tool_choice")
    return {
        "system_text": system_text,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": stops,
        "json_mode": json_mode,
        "stream": body.get("stream") is True,
        "temperature": body.get("temperature"),
        "tool_choice": tool_choice,
    }


def openai_messages(system_text, messages):
    out = []
    if system_text:
        out.append({"role": "system", "content": system_text})
    for role, blocks in messages:
        if role == "system":
            continue
        has_image = any("image_b64" in b or "image_url" in b for b in blocks)
        if has_image:
            content = []
            for block in blocks:
                if "text" in block:
                    content.append({"type": "text", "text": block["text"]})
                elif "image_b64" in block:
                    mime, data = block["image_b64"]
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:%s;base64,%s" % (mime, data)},
                        }
                    )
                elif "image_url" in block:
                    content.append(
                        {"type": "image_url", "image_url": {"url": block["image_url"]}}
                    )
            out.append({"role": role, "content": content})
        else:
            text = "\n".join(b["text"] for b in blocks if "text" in b)
            out.append({"role": role, "content": text})
    return out


def anthropic_messages(messages):
    out = []
    for role, blocks in messages:
        if role == "system":
            continue
        parts = []
        for block in blocks:
            if "text" in block:
                parts.append({"type": "text", "text": block["text"]})
            elif "image_b64" in block:
                mime, data = block["image_b64"]
                parts.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": data},
                    }
                )
            elif "image_url" in block:
                parts.append(
                    {"type": "image", "source": {"type": "url", "url": block["image_url"]}}
                )
        if len(parts) == 1 and parts[0]["type"] == "text":
            out.append({"role": role, "content": parts[0]["text"]})
        else:
            out.append({"role": role, "content": parts})
    return out


def gemini_contents(messages):
    contents = []
    for role, blocks in messages:
        if role == "system":
            continue
        parts = []
        for block in blocks:
            if "text" in block:
                parts.append({"text": block["text"]})
            elif "image_b64" in block:
                mime, data = block["image_b64"]
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            elif "image_url" in block:
                parts.append({"fileData": {"fileUri": block["image_url"]}})
        contents.append(
            {"role": "model" if role == "assistant" else "user", "parts": parts}
        )
    return contents


def provider_body(provider, model, req, stream):
    """Render the neutral request in one provider's wire. Returns (body, path)."""
    wire = provider.get("wire")
    if wire == "openai":
        body = {
            "model": model,
            "messages": openai_messages(req["system_text"], req["messages"]),
            "stream": stream,
        }
        if req["max_tokens"] is not None:
            body["max_tokens"] = req["max_tokens"]
        if req["temperature"] is not None:
            body["temperature"] = req["temperature"]
        if req["tools"]:
            body["tools"] = [
                {
                    "type": "function",
                    "function": strip_none(
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"],
                        }
                    ),
                }
                for tool in req["tools"]
            ]
        if req["tool_choice"] is not None:
            body["tool_choice"] = req["tool_choice"]
        if req["stops"]:
            body["stop"] = req["stops"]
        if req["json_mode"]:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body, "/v1/chat/completions"
    if wire == "anthropic":
        body = {
            "model": model,
            "messages": anthropic_messages(req["messages"]),
            "stream": stream,
        }
        if req["system_text"]:
            body["system"] = req["system_text"]
        if req["max_tokens"] is not None:
            body["max_tokens"] = req["max_tokens"]
        if req["temperature"] is not None:
            body["temperature"] = req["temperature"]
        if req["tools"]:
            body["tools"] = [
                strip_none(
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool["parameters"],
                    }
                )
                for tool in req["tools"]
            ]
        if req["tool_choice"] is not None:
            body["tool_choice"] = req["tool_choice"]
        if req["stops"]:
            body["stop_sequences"] = req["stops"]
        return body, "/v1/messages"
    body = {"contents": gemini_contents(req["messages"])}
    if req["system_text"]:
        body["systemInstruction"] = {"parts": [{"text": req["system_text"]}]}
    if req["tools"]:
        body["tools"] = [
            {
                "functionDeclarations": [
                    strip_none(
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"],
                        }
                    )
                    for tool in req["tools"]
                ]
            }
        ]
    config = {}
    if req["max_tokens"] is not None:
        config["maxOutputTokens"] = req["max_tokens"]
    if req["stops"]:
        config["stopSequences"] = req["stops"]
    if req["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    action = "streamGenerateContent" if stream else "generateContent"
    path = "/v1beta/models/%s:%s" % (
        urllib.parse.quote(model, safe=""),
        action,
    )
    if stream:
        path += "?alt=sse"
    return body, path


# ---------------------------------------------------------------------------
# Provider answers: every wire back into one neutral result
# {ok, text, input, output, stop}. The text is the provider's answer whole;
# the client checks compare it and the usage counts against the provider.
# ---------------------------------------------------------------------------


def split_sse(text):
    """Split a server-sent-events body into (event, data) frames."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for chunk in normalized.split("\n\n"):
        if not chunk.strip():
            continue
        event = None
        data = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def text_from_blocks(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def parse_provider_json(wire, text):
    try:
        doc = json.loads(text)
    except ValueError:
        return {"ok": False, "status": 0}
    if not isinstance(doc, dict):
        return {"ok": False, "status": 0}
    if wire == "openai":
        choices = doc.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return {"ok": False, "status": 0}
        choice = choices[0]
        message = choice.get("message") or {}
        usage = doc.get("usage") or {}
        return {
            "ok": True,
            "text": text_from_blocks(message.get("content")),
            "input": usage.get("prompt_tokens"),
            "output": usage.get("completion_tokens"),
            "stop": choice.get("finish_reason") or "stop",
        }
    if wire == "anthropic":
        usage = doc.get("usage") or {}
        return {
            "ok": True,
            "text": text_from_blocks(doc.get("content")),
            "input": usage.get("input_tokens"),
            "output": usage.get("output_tokens"),
            "stop": doc.get("stop_reason") or "end_turn",
        }
    candidates = doc.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return {"ok": False, "status": 0}
    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text_out = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )
    usage = doc.get("usageMetadata") or {}
    return {
        "ok": True,
        "text": text_out,
        "input": usage.get("promptTokenCount"),
        "output": usage.get("candidatesTokenCount"),
        "stop": candidate.get("finishReason") or "STOP",
    }


def parse_provider_stream(wire, text):
    pieces = []
    input_tokens = None
    output_tokens = None
    stop = None
    for event, data in split_sse(text):
        if data.strip() == "[DONE]":
            continue
        try:
            doc = json.loads(data)
        except ValueError:
            continue
        if not isinstance(doc, dict):
            continue
        if wire == "openai":
            usage = doc.get("usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
            choices = doc.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                pieces.append(delta["content"])
            if isinstance(choice.get("finish_reason"), str):
                stop = choice["finish_reason"]
        elif wire == "anthropic":
            if event == "message_start":
                message = doc.get("message") or {}
                usage = message.get("usage") or {}
                input_tokens = usage.get("input_tokens")
            elif event == "content_block_delta":
                delta = doc.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(
                    delta.get("text"), str
                ):
                    pieces.append(delta["text"])
            elif event == "message_delta":
                delta = doc.get("delta") or {}
                if isinstance(delta.get("stop_reason"), str):
                    stop = delta["stop_reason"]
                usage = doc.get("usage") or {}
                if usage.get("output_tokens") is not None:
                    output_tokens = usage.get("output_tokens")
        else:
            candidates = doc.get("candidates") or []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        pieces.append(part["text"])
                if isinstance(candidate.get("finishReason"), str):
                    stop = candidate["finishReason"]
            usage = doc.get("usageMetadata")
            if isinstance(usage, dict):
                input_tokens = usage.get("promptTokenCount")
                output_tokens = usage.get("candidatesTokenCount")
    joined = "".join(pieces)
    if not joined:
        return {"ok": False, "status": 0}
    return {
        "ok": True,
        "text": joined,
        "input": input_tokens,
        "output": output_tokens,
        "stop": stop or WIRE_STOPS[wire],
    }


def call_provider(env_url, provider, model, req, stream):
    """Send one request to one provider; a refusal comes back not-ok."""
    body, path = provider_body(provider, model, req, stream)
    url = env_url.rstrip("/") + provider.get("path", "") + path
    headers = auth_headers(provider)
    headers["Content-Type"] = "application/json"
    if provider.get("wire") == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, raw = http_call("POST", url, headers, body, timeout=15)
    if status != 200:
        return {"ok": False, "status": status}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "status": 0}
    if stream:
        return parse_provider_stream(provider.get("wire"), text)
    return parse_provider_json(provider.get("wire"), text)


# ---------------------------------------------------------------------------
# Routing: best free model first, fail over down the rating order.
# ---------------------------------------------------------------------------


def meets_requirements(row, requires):
    if requires.get("tools") and not row.get("tools"):
        return False
    if requires.get("vision") and not row.get("vision"):
        return False
    if requires.get("json") and not row.get("json"):
        return False
    minimum = requires.get("min_context")
    if minimum is not None and (row.get("context_length") or 0) < minimum:
        return False
    return True


class Engine:
    """Holds the brief, the catalog and the per-trial provider state."""

    def __init__(self, env_url, brief):
        self.env_url = env_url
        self.providers = brief["providers"]
        self.ratings = brief["ratings"]
        self.manifest = brief["manifest"]
        self.by_id = {p.get("id"): p for p in self.providers}
        self.catalog = None
        self.catalog_text = None
        self.served_count = 0
        self.attempts = {}
        self.refusals = {}
        self.dead = set()

    def build_catalog(self):
        self.catalog = build_catalog(self.env_url, self.providers, self.ratings)
        self.catalog_text = json.dumps(self.catalog, separators=(",", ":"))
        return self.catalog

    def free_model_ids(self):
        ids = []
        if self.catalog is None:
            return ids
        for entry in self.catalog["providers"]:
            for model in entry["models"]:
                if model.get("free") is True:
                    ids.append(model["id"])
        return ids

    def register(self, port):
        body = {
            "protocol_version": PROTOCOL_VERSION,
            "port": port,
            "catalog": self.catalog,
        }
        url = self.env_url.rstrip("/") + "/register"
        for _ in range(20):
            status, raw = http_call(
                "POST", url, {"Content-Type": "application/json"}, body, timeout=10
            )
            if status == 200:
                try:
                    return json.loads(raw.decode("utf-8")).get("ok") is True
                except (ValueError, UnicodeDecodeError):
                    return False
            time.sleep(0.5)
        return False

    def work_state(self):
        status, raw = http_call(
            "GET", self.env_url.rstrip("/") + "/work", {}, timeout=5
        )
        if status != 200:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def requirements_for(self, api, req):
        """The manifest entry for the request now being sent, if known."""
        request_id = None
        state = self.work_state()
        if isinstance(state, dict):
            request_id = state.get("next")
        if not isinstance(request_id, str) or request_id not in self.manifest:
            ids = sorted(self.manifest.keys())
            if 0 <= self.served_count < len(ids):
                request_id = ids[self.served_count]
            else:
                request_id = None
        self.served_count += 1
        if request_id is not None:
            return dict(self.manifest[request_id].get("requires") or {})
        return self.infer_requirements(req)

    def infer_requirements(self, req):
        """Fallback when the manifest cannot name the request: read the body."""
        requires = {}
        if req["tools"]:
            requires["tools"] = True
        for _, blocks in req["messages"]:
            if any("image_b64" in b or "image_url" in b for b in blocks):
                requires["vision"] = True
        if req["json_mode"]:
            requires["json"] = True
        chars = 0
        for _, blocks in req["messages"]:
            for block in blocks:
                chars += len(block.get("text", ""))
        if chars > 50000:
            requires["min_context"] = 200000
        return requires

    def ranked_candidates(self, requires, client_api, json_mode):
        """Free models meeting the request, best rating first.

        Providers tied on rating are tried least-used first, so the load
        spreads across equal-rated providers instead of hammering one of
        them until its budget or a scheduled refusal knocks it out. The
        order is a function of the trial's own history, so it replays.
        """
        found = []
        for row in self.ratings:
            if row.get("free") is not True:
                continue
            if not meets_requirements(row, requires):
                continue
            provider = self.by_id.get(row.get("provider"))
            if provider is None:
                continue
            # An Anthropic-wire provider cannot carry an OpenAI JSON-mode
            # request: its wire has no response_format to echo back.
            if client_api == "openai" and json_mode and provider.get("wire") == "anthropic":
                continue
            found.append(
                (
                    -row.get("rating_milli", 0),
                    self.attempts.get(row.get("provider"), 0),
                    row.get("provider"),
                    row.get("model"),
                    provider,
                )
            )
        found.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [(provider, model) for _, _, _, model, provider in found]

    def answer(self, api, req, stream):
        """Try candidates in rating order until one provider answers."""
        requires = self.requirements_for(api, req)
        candidates = self.ranked_candidates(requires, api, req["json_mode"])
        for _ in range(2):
            refused = set()
            for provider, model in candidates:
                provider_id = provider.get("id")
                if provider_id in self.dead or provider_id in refused:
                    continue
                self.attempts[provider_id] = self.attempts.get(provider_id, 0) + 1
                result = call_provider(self.env_url, provider, model, req, stream)
                if result.get("ok"):
                    self.refusals[provider_id] = 0
                    return model, result
                refused.add(provider_id)
                count = self.refusals.get(provider_id, 0) + 1
                self.refusals[provider_id] = count
                # Three refusals in a row means the budget is spent or the
                # provider is out: stop spending calls on it this trial. A
                # lone scheduled refusal never reaches three on its own.
                if count >= 3:
                    self.dead.add(provider_id)
        return None


# ---------------------------------------------------------------------------
# Client answers: the provider's text and usage, in the client's wire.
# ---------------------------------------------------------------------------


def split_text(text, size=13):
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


def sse_data(payload):
    return "data: %s\n\n" % json.dumps(payload, separators=(",", ":"))


def openai_chunk(model, choices, usage=None):
    """One chat-completion chunk envelope; usage only on the final frame."""
    chunk = {
        "id": "chatcmpl-sim",
        "object": "chat.completion.chunk",
        "created": CREATED,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def openai_response(model, result, stream):
    finish = stop_in_wire(result["stop"], "openai")
    text = result["text"]
    usage = {
        "prompt_tokens": result["input"],
        "completion_tokens": result["output"],
        "total_tokens": (result["input"] or 0) + (result["output"] or 0),
    }
    if not stream:
        body = {
            "id": "chatcmpl-sim",
            "object": "chat.completion",
            "created": CREATED,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                    "logprobs": None,
                }
            ],
            "usage": usage,
        }
        payload = (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")
        return 200, {"Content-Type": "application/json"}, payload
    frames = []
    frames.append(
        sse_data(
            openai_chunk(
                model,
                [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            )
        )
    )
    for piece in split_text(text):
        frames.append(
            sse_data(
                openai_chunk(
                    model,
                    [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                )
            )
        )
    frames.append(
        sse_data(
            openai_chunk(
                model,
                [{"index": 0, "delta": {}, "finish_reason": finish}],
            )
        )
    )
    frames.append(
        sse_data(
            openai_chunk(model, [], usage),
        )
    )
    frames.append("data: [DONE]\n\n")
    payload = "".join(frames).encode("utf-8")
    return 200, {"Content-Type": "text/event-stream; charset=utf-8"}, payload


def anthropic_response(model, result, stream):
    finish = stop_in_wire(result["stop"], "anthropic")
    text = result["text"]
    if not stream:
        body = {
            "id": "msg_sim",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": finish,
            "stop_sequence": None,
            "usage": {
                "input_tokens": result["input"],
                "output_tokens": result["output"],
            },
        }
        payload = (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")
        return 200, {"Content-Type": "application/json"}, payload
    frames = []

    def event(name, payload):
        frames.append(
            "event: %s\n%s" % (name, sse_data(payload))
        )

    event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_sim",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": result["input"], "output_tokens": 0},
            },
        },
    )
    event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    for piece in split_text(text):
        event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            },
        )
    event("content_block_stop", {"type": "content_block_stop", "index": 0})
    event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": finish, "stop_sequence": None},
            "usage": {"output_tokens": result["output"]},
        },
    )
    event("message_stop", {"type": "message_stop"})
    payload = "".join(frames).encode("utf-8")
    return 200, {"Content-Type": "text/event-stream; charset=utf-8"}, payload


# ---------------------------------------------------------------------------
# HTTP server: the two client routes plus the model list.
# ---------------------------------------------------------------------------

ENGINE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    daemon_threads = True

    def log_message(self, *args):
        pass

    def _send(self, status, headers, payload):
        headers = dict(headers)
        headers["Content-Length"] = str(len(payload))
        try:
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            if payload:
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, status, doc):
        payload = (json.dumps(doc, separators=(",", ":")) + "\n").encode("utf-8")
        self._send(status, {"Content-Type": "application/json"}, payload)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path != "/v1/models":
            self._send_json(404, {"error": "not_found"})
            return
        ids = ENGINE.free_model_ids()
        if self.headers.get("x-api-key") is not None:
            body = {
                "data": [
                    {
                        "type": "model",
                        "id": model_id,
                        "display_name": model_id,
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                    for model_id in ids
                ],
                "has_more": False,
            }
        else:
            body = {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": CREATED,
                        "owned_by": "free-model-aggregator-01",
                    }
                    for model_id in ids
                ],
            }
        self._send_json(200, body)

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "bad_request"})
            return
        if not isinstance(body, dict):
            self._send_json(400, {"error": "bad_request"})
            return
        stream = body.get("stream") is True
        req = normalize_request(body, api)
        answered = ENGINE.answer(api, req, stream)
        if answered is None:
            self._send_json(502, {"error": "no_provider_available"})
            return
        model, result = answered
        if api == "openai":
            status, headers, payload = openai_response(model, result, stream)
        else:
            status, headers, payload = anthropic_response(model, result, stream)
        self._send(status, headers, payload)


def free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def main(argv):
    global ENGINE
    list_mode = "--list" in argv
    prompt_path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    brief = load_brief(prompt_path)
    if list_mode:
        if brief and env_url:
            catalog = build_catalog(env_url, brief["providers"], brief["ratings"])
            sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        else:
            sys.stdout.write('{"providers":[]}\n')
        return 0
    if not brief or not env_url:
        sys.stderr.write("free-model-aggregator-01: needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL\n")
        return 1
    ENGINE = Engine(env_url, brief)
    ENGINE.build_catalog()
    port = free_port()
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    thread.start()
    if not ENGINE.register(port):
        server.shutdown()
        server.server_close()
        sys.stderr.write("free-model-aggregator-01: register failed\n")
        return 1
    while True:
        state = ENGINE.work_state()
        if isinstance(state, dict) and state.get("done") is True:
            break
        time.sleep(0.25)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
