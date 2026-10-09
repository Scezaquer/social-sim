"""OpenAI-compatible stub for the SiliSocS runner tests.

GET /v1/models; POST /v1/chat/completions. With `tools` it returns ONE tool call chosen
at random among the offered tools, with arguments generated from the JSON schema
(integer ids are drawn from 'Tweet ID: <n>' occurrences in the prompt, else 1..50).
Without tools it returns text in SiliSocS's 'ACTION TYPE / TARGET ID / CONTENT' format.
Model 'broken' -> HTTP 500. Every request is appended to $STUB_LOG (if set).
Usage: python stub_server.py PORT
"""
import hashlib, json, os, random, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["base", "lora0", "lora1", "broken"]
WORDS = "honestly think the new transit plan could work but costs matter more".split()


def _text(rng):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 12))).capitalize() + "."


def _args(schema, rng, ids):
    props = (schema or {}).get("properties", {}) or {}
    out = {}
    for key, spec in props.items():
        typ = spec.get("type")
        if typ == "integer" or key.endswith("_id"):
            out[key] = rng.choice(ids) if ids else rng.randint(1, 50)
        elif typ == "number":
            out[key] = rng.random()
        elif typ == "boolean":
            out[key] = False
        elif key in (schema.get("required") or []) or key in ("status", "content", "text"):
            out[key] = _text(rng)
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"object": "list",
                                    "data": [{"id": m, "object": "model"} for m in MODELS]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if os.environ.get("STUB_LOG"):
            with open(os.environ["STUB_LOG"], "a") as f:
                f.write(json.dumps({k: req.get(k) for k in
                                    ("model", "max_tokens", "temperature", "tool_choice")}
                                   | {"tools": [t["function"]["name"] for t in req.get("tools") or []],
                                      "n_chars": sum(len(str(m.get("content"))) for m in req.get("messages", []))}) + "\n")
        if req.get("model") == "broken":
            return self._send(500, {"error": {"message": "stub failure"}})
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "only chat.completions"}})
        time.sleep(0.05)
        prompt = "\n".join(str(m.get("content")) for m in req.get("messages", []))
        rng = random.Random(int(hashlib.md5(prompt.encode()).hexdigest(), 16))
        ids = [int(x) for x in re.findall(r"Tweet ID: (\d+)", prompt)]
        tools = req.get("tools") or []
        msg = {"role": "assistant", "content": None}
        if tools:
            fn = rng.choice(tools)["function"]
            msg["tool_calls"] = [{"id": f"call_{rng.randint(0, 10**9)}", "type": "function",
                                  "function": {"name": fn["name"], "arguments": json.dumps(
                                      _args(fn.get("parameters"), rng, ids))}}]
            finish = "tool_calls"
        else:
            kind = rng.choice(["POST", "REPLY", "LIKE"]) if ids else "POST"
            msg["content"] = (f"FINAL DECISION:\nACTION TYPE: {kind}\n"
                              f"TARGET ID: {rng.choice(ids) if kind != 'POST' else ''}\n"
                              f"CONTENT: {_text(rng)}\nREASONING: stub")
            finish = "stop"
        self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion",
                         "created": int(time.time()), "model": req["model"],
                         "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                         "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 20,
                                   "total_tokens": len(prompt) // 4 + 20}})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
