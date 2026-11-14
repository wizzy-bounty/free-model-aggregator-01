# Free Model Aggregator

One server that answers the OpenAI Chat Completions API and the Anthropic
Messages API — in JSON and in streaming — by routing every request to the
best free model across ten simulated free-tier providers.

## How it works

- The server reads its brief (`MAKEFUN_PROMPT_FILE`): the ten providers,
  the sealed quality table and the workload manifest.
- At startup it asks each provider for its own model list and builds one
  catalog document from the answers, then registers its port and that
  catalog with the trial env.
- For each incoming request it looks up what the request needs (tools,
  vision, JSON mode, minimum context) and picks the highest-rated free
  model that meets those needs. A provider that refuses with 402 or 429
  costs nothing but a retry: the next best candidate serves the request.
- Requests are translated into the chosen provider's own wire — OpenAI,
  Anthropic or Gemini — keeping the system prompt, user text, images,
  tools, stop sequences, `max_tokens` and JSON mode exactly as the client
  sent them, so the provider echoes them back unchanged. Answers come back
  in the wire the client used, with the provider's text whole and its usage
  counts as reported.
- `GET /v1/models` answers in the wire the auth header names: a Bearer
  token gets the OpenAI list, an `x-api-key` gets the Anthropic list, both
  listing exactly the free models in the registered catalog.
- When the env says the workload is done, the server exits 0.

## Running it

```sh
python3 main.py            # serve a trial (needs MAKEFUN_PROMPT_FILE
                           # and MAKEFUN_ENV_URL from the trial env)
python3 main.py --list     # print the catalog document and exit 0
```

Python 3.13, standard library only — nothing to install. Every routing
choice is a function of the brief, the requests and the providers' answers,
so a replay of the same trial leaves the same state export.

## Layout

- `main.py` — the whole server.
- `.makefun/owner.json` — the claim file naming the payout wallet.
- `LICENSE` — MIT.
