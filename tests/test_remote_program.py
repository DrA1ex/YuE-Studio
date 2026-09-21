"""Remote graph cache migration without loading weights or connecting to a phone."""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from yue2.remote import client
from yue2.ane.mil import build_program


class RemoteProgramTests(unittest.TestCase):
    def make_client(self, programs):
        c = client.RemoteClient.__new__(client.RemoteClient)
        c.info = {"weights": "keep-existing-weights", "programs": programs[:]}
        c.lock = threading.RLock()
        c.conn = Mock()
        return c

    def test_old_phone_and_disk_cache_trigger_new_program_not_weight_transfer(self):
        old = "layers2_1024_2048_plain_q512_k1024_v7"
        remote = self.make_client([old])
        config = SimpleNamespace(hidden_size=2048, num_attention_heads=16, num_key_value_heads=8,
                                 head_dim=128, intermediate_size=6144, rms_norm_eps=1e-6)
        def generate(*args, package_path, **kwargs):
            self.assertEqual(kwargs["attention_mode"], "softmax")
            self.assertEqual(kwargs["qblk"], 128)
            package_path.mkdir()
            (package_path / "model.mlmodel").write_bytes(b"new-model")
        with tempfile.TemporaryDirectory() as directory, patch.object(client, "CACHE", Path(directory)), patch.object(client, "build_program", side_effect=generate) as builder:
            legacy = Path(directory) / (old + ".mlpackage")
            legacy.mkdir()
            (legacy / "model.mlmodel").write_bytes(b"old-model")
            name = remote.ensure_program(1024, 2048, config)
            self.assertNotEqual(name, old)
            builder.assert_called_once()
            self.assertEqual(remote.conn.call.call_count, 1)
            args, kwargs = remote.conn.call.call_args
            self.assertEqual(args, ("program", b"new-model"))
            self.assertEqual(kwargs["name"], name)
            self.assertEqual(remote.info["weights"], "keep-existing-weights")
            self.assertEqual((legacy / "model.mlmodel").read_bytes(), b"old-model")
            remote.ensure_program(1024, 2048, config)
            self.assertEqual(remote.conn.call.call_count, 1)

    def test_updated_phone_cache_needs_no_regeneration(self):
        name = client.program_name(1536, 2048)
        remote = self.make_client([name])
        with patch.object(client, "build_program") as builder:
            self.assertEqual(remote.ensure_program(1536, 2048, None), name)
            builder.assert_not_called()
            remote.conn.call.assert_not_called()

    def test_invalid_attention_mode_fails_before_building_a_package(self):
        with self.assertRaises(ValueError):
            build_program([None], None, S=512, P=1024, attention_mode="typo")


if __name__ == "__main__":
    unittest.main()
