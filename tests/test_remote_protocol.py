"""Remote protocol framing without opening a real socket."""
import json
import struct
import unittest

from yue2.remote.protocol import Connection


class FakeSocket:
    def __init__(self, incoming=b""):
        self.incoming = bytearray(incoming)
        self.sent = bytearray()

    def sendall(self, data):
        self.sent.extend(data)

    def recv_into(self, view, size):
        if not self.incoming:
            return 0
        count = min(size, len(self.incoming), 3)
        view[:count] = self.incoming[:count]
        del self.incoming[:count]
        return count


def encode_frame(header, payload=b""):
    blob = json.dumps(header).encode()
    return struct.pack(">I", len(blob)) + blob + struct.pack(">Q", len(payload)) + payload


class RemoteProtocolTests(unittest.TestCase):
    def test_call_parts_streams_existing_buffers_as_one_payload(self):
        sock = FakeSocket(encode_frame({"op": "weights_layer", "ok": True}))
        conn = Connection.__new__(Connection)
        conn.sock = sock
        conn.host, conn.port = "test", 1

        words = memoryview(bytearray(b"cdef")).cast("H")
        header, payload = conn.call_parts(
            "weights_layer",
            [b"ab", words],
            identity="weights",
            layer=3,
        )

        self.assertTrue(header["ok"])
        self.assertEqual(payload, b"")

        sent = bytes(sock.sent)
        header_len = struct.unpack(">I", sent[:4])[0]
        request = json.loads(sent[4:4 + header_len])
        payload_len_offset = 4 + header_len
        payload_len = struct.unpack(">Q", sent[payload_len_offset:payload_len_offset + 8])[0]
        wire_payload = sent[payload_len_offset + 8:]

        self.assertEqual(request, {"identity": "weights", "layer": 3, "op": "weights_layer"})
        self.assertEqual(payload_len, 6)
        self.assertEqual(wire_payload, b"abcdef")


if __name__ == "__main__":
    unittest.main()
