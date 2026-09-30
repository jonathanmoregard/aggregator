"""A stand-in for ``llama-server --embedding`` on a real loopback socket.

Real HTTP, not a patched ``urlopen``: what the ``server`` backend has to get
right is the wire — which endpoint it asks, what it sends, how it reads the
answer back, and what it does when the socket is gone — and a mocked opener
would let every one of those be wrong while the test passes.

The two endpoints are the ones the backend uses, answering in the shape
llama.cpp build 10273 was observed to answer on this machine (``GET
/v1/models`` and ``POST /v1/embeddings``). The vectors are a deterministic
function of the input text, so a test can say which vector belongs to which
input without trusting the order the response came back in.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SERVED_FILE = "Qwen3-Embedding-0.6B-Q8_0.gguf"
NATIVE_DIM = 1024


def vector_for(text: str, dim: int = NATIVE_DIM) -> np.ndarray:
    """The raw (un-normalised, native-width) vector the stub returns for ``text``."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32) * 3


class EmbedServerStub:
    """Run with ``with EmbedServerStub() as stub: ... stub.url ...``.

    Knobs, all plain attributes so a test can change them mid-flight:

    * ``model_path`` / ``n_embd`` / ``ftype`` — what ``/v1/models`` claims.
    * ``loading_replies`` — answer this many requests with llama-server's
      503 "Loading model" before behaving.
    * ``die_after`` — serve this many embedding requests, then stop listening
      altogether (the unit being stopped under a running worker).
    * ``shuffle`` — return ``data`` in reverse order, which the OpenAI shape
      allows; the client must put rows back by ``index``.
    """

    def __init__(
        self,
        model_path: str = f"/models/{SERVED_FILE}",
        n_embd: int = NATIVE_DIM,
        ftype: str | None = "Q8_0",
    ) -> None:
        self.model_path = model_path
        self.n_embd = n_embd
        self.ftype = ftype
        self.loading_replies = 0
        self.die_after: int | None = None
        self.shuffle = True
        #: Set once ``die_after`` is reached. From then on every connection is
        #: dropped without a reply — deterministic, unlike racing the serve
        #: loop's own shutdown poll.
        self.dead = False
        #: Every ``input`` list POSTed to /v1/embeddings, in arrival order.
        self.embed_requests: list[list[str]] = []
        self.model_requests = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> EmbedServerStub:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output clean
                pass

            def _reply(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _dropped(self) -> bool:
                if stub.dead:
                    self.close_connection = True
                    return True
                return False

            def _loading(self) -> bool:
                if stub.loading_replies > 0:
                    stub.loading_replies -= 1
                    self._reply(
                        503,
                        {
                            "error": {
                                "code": 503,
                                "message": "Loading model",
                                "type": "unavailable_error",
                            }
                        },
                    )
                    return True
                return False

            def do_GET(self):  # noqa: N802 - http.server's spelling
                if self._dropped() or self._loading():
                    return
                if self.path != "/v1/models":
                    self._reply(404, {"error": {"message": "not found"}})
                    return
                stub.model_requests += 1
                meta = {"n_embd": stub.n_embd, "n_ctx": 8192}
                if stub.ftype is not None:
                    meta["ftype"] = stub.ftype
                self._reply(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": stub.model_path,
                                "aliases": [stub.model_path],
                                "object": "model",
                                "owned_by": "llamacpp",
                                "meta": meta,
                            }
                        ],
                    },
                )

            def do_POST(self):  # noqa: N802
                if self._dropped() or self._loading():
                    return
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length) or b"{}")
                if self.path != "/v1/embeddings":
                    self._reply(404, {"error": {"message": "not found"}})
                    return
                inputs = req["input"]
                if isinstance(inputs, str):
                    inputs = [inputs]
                stub.embed_requests.append(list(inputs))
                data = [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": vector_for(t, stub.n_embd).tolist(),
                    }
                    for i, t in enumerate(inputs)
                ]
                if stub.shuffle:
                    data.reverse()
                self._reply(200, {"object": "list", "data": data})
                if stub.die_after is not None and len(stub.embed_requests) >= stub.die_after:
                    stub.dead = True
                    # And stop listening. From another thread: shutdown()
                    # blocks until the serve loop exits, and this handler is
                    # running inside it.
                    threading.Thread(target=stub.stop, daemon=True).start()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            httpd, self._httpd = self._httpd, None
            httpd.shutdown()
            httpd.server_close()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def url(self) -> str:
        assert self._httpd is not None
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"


def closed_port_url() -> str:
    """A loopback URL nothing is listening on — the unit stopped.

    Bind, read the port, close: the kernel will refuse connections to it
    until something else claims it, which in a test run is effectively never.
    """
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"
