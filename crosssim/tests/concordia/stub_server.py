"""Minimal OpenAI-compatible stub: GET /v1/models, POST /v1/completions.
Usage: python stub_server.py PORT   (model name 'broken' -> HTTP 500)."""
import json, random, sys, time, hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["base", "lora0", "lora1", "broken"]
WORDS = "honestly think the new transit plan could work but costs matter more".split()

class H(BaseHTTPRequestHandler):
  def log_message(self, *a): pass
  def _send(self, code, obj):
    b = json.dumps(obj).encode(); self.send_response(code)
    self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
    self.end_headers(); self.wfile.write(b)
  def do_GET(self):
    if self.path.rstrip("/").endswith("/models"):
      return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
    self._send(404, {"error": "not found"})
  def do_POST(self):
    req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    if req.get("model") == "broken":
      return self._send(500, {"error": {"message": "stub failure"}})
    time.sleep(0.05)
    h = int(hashlib.md5(req["prompt"].encode()).hexdigest(), 16)
    rng = random.Random(h)
    text = " " + " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 12))) + ".\nQuestion: junk after stop"
    stop = req.get("stop") or []
    for s in stop:
      if s in text: text = text[:text.index(s)]
    self._send(200, {"id": "cmpl-stub", "object": "text_completion", "created": int(time.time()),
      "model": req["model"], "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": len(req["prompt"]) // 4, "completion_tokens": 10, "total_tokens": 10}})

if __name__ == "__main__":
  ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
