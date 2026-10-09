"""Test-only stand-in for `vllm serve`: chat (tool calls) + completions (text and prompt_logprobs)."""
import hashlib, json, random, sys, time, zlib, importlib.util, os
from http.server import ThreadingHTTPServer
spec = importlib.util.spec_from_file_location("st", os.path.join(os.path.dirname(__file__), "oasis", "stub_server.py"))
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
st.MODELS = ["base"] + [f"lora{i}" for i in range(25)]

class H(st.H):
    def do_POST(self):
        if not self.path.rstrip("/").endswith("/completions") or self.path.rstrip("/").endswith("/chat/completions"):
            return super().do_POST()
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = req["prompt"]
        rng = random.Random(zlib.crc32(json.dumps([req["model"], p]).encode()))
        ch = {"index": 0, "text": st._text(rng), "finish_reason": "stop", "logprobs": None}
        if req.get("prompt_logprobs") is not None:
            ids = p if isinstance(p, list) else list(range(len(p.split())))
            ch["prompt_logprobs"] = [None] + [{str(t): {"logprob": -rng.random() * 3, "rank": 1}} for t in ids[1:]]
        self._send(200, {"choices": [ch]})

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
