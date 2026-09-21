"""Wire format shared with the iPhone companion (app/YuERemote/Sources/Protocol.swift).

Frame: u32 header length | header JSON | u64 payload length | payload. Each request gets one
terminal reply carrying the same "op"; long operations send {"op": "progress", "text": ...}
frames first.
"""
from __future__ import annotations
import json, socket, struct


class RemoteError(RuntimeError):
    pass


class Connection:
    def __init__(self, host, port, timeout=1800.0):
        self.sock = socket.create_connection((host, int(port)), timeout=20)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.host, self.port = host, int(port)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def send(self, header, payload=b""):
        blob = json.dumps(header).encode()
        self.sock.sendall(struct.pack(">I", len(blob)) + blob + struct.pack(">Q", len(payload)))
        if len(payload):
            self.sock.sendall(payload)

    def _recv_exactly(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], min(n - got, 4 << 20))
            if k == 0:
                raise RemoteError("the iPhone closed the connection")
            got += k
        return bytes(buf)

    def recv(self):
        (hl,) = struct.unpack(">I", self._recv_exactly(4))
        header = json.loads(self._recv_exactly(hl))
        (pl,) = struct.unpack(">Q", self._recv_exactly(8))
        payload = self._recv_exactly(pl) if pl else b""
        return header, payload

    def call(self, op, payload=b"", on_progress=None, **fields):
        """Send one request and wait for its reply, relaying progress frames."""
        self.send(dict(fields, op=op), payload)
        while True:
            header, data = self.recv()
            if header.get("op") == "progress":
                if on_progress is not None:
                    on_progress(header.get("text", ""))
                continue
            if header.get("op") != op:
                raise RemoteError(f"expected a {op} reply, got {header.get('op')}")
            if not header.get("ok"):
                raise RemoteError(header.get("error", f"{op} failed on the iPhone"))
            return header, data
