"""Synthetic vocal-window tests; separator weights and UI are not used."""
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from voice_activity import activity_windows, download_separation_model, separation_model_available, separation_asset_path


class VoiceActivityTests(unittest.TestCase):
    def test_separator_download_uses_torch_downloader_and_validates_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch("voice_activity.SEPARATION_MIN_BYTES", 4), patch("torch.hub.get_dir", return_value=tmp):
            def fake_download(url, destination, progress):
                self.assertTrue(url.endswith("models/hdemucs_high_musdbhq_only.pt"))
                self.assertTrue(progress)
                Path(destination).write_bytes(b"1234")

            with patch("torch.hub.download_url_to_file", side_effect=fake_download) as downloader:
                path = download_separation_model()
            self.assertTrue(path.is_file())
            downloader.assert_called_once()

    def test_partial_separator_asset_is_not_reported_as_installed(self):
        with tempfile.TemporaryDirectory() as tmp, patch("voice_activity.SEPARATION_MIN_BYTES", 4), patch("torch.hub.get_dir", return_value=tmp):
            path = separation_asset_path()
            path.parent.mkdir(parents=True)
            path.write_bytes(b"123")
            self.assertFalse(separation_model_available())
            path.write_bytes(b"1234")
            self.assertTrue(separation_model_available())

    def test_activity_merges_short_gaps_and_bounds_long_regions(self):
        scores = [0.0] * 4 + [1.0] * 70 + [0.0] + [1.0] * 70
        windows = activity_windows(scores, .5, len(scores) * .5, maximum=10.0)
        self.assertTrue(windows)
        self.assertTrue(all(end - start <= 10.0 for start, end in windows))
        self.assertLessEqual(windows[0][0], 1.5)

    def test_empty_or_silent_activity_falls_back_to_bounded_full_audio(self):
        windows = activity_windows([0.0] * 20, .5, 10.0, maximum=4.0)
        self.assertEqual(windows, [(0.0, 4.0), (2.0, 6.0), (4.0, 8.0), (6.0, 10.0)])


if __name__ == "__main__":
    unittest.main()
