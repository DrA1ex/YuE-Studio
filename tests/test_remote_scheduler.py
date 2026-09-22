"""Exercise actual scheduling without models, sockets, inference or UI automation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('remote_scheduler_worker', ROOT / 'tools/yue2_worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class RemoteSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(w, 'REMOTE', object()), patch.object(w, 'emit')]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.sched = w.Scheduler()

    def songs(self, count, ane=False, frames=100):
        songs = [w.Song('run', i, i, SimpleNamespace(cot='off', abc=None),
                        ROOT / 'unused' / str(i), 'full', 32, 100) for i in range(count)]
        for song in songs:
            song.state = w.SYNTH_WAIT
            song.codec = [0] * frames
            song.wants_ane = ane
        with patch.object(self.sched, 'tick'):
            self.sched.submit(songs)
        return songs

    def test_balanced_whole_song_execution_one_through_eight(self):
        for concurrent in (False, True):
            for ane in (False, True):
                for count in range(1, 9):
                    with self.subTest(concurrent=concurrent, ane=ane, count=count), patch.object(w, 'CONCURRENT', concurrent):
                        self.sched = w.Scheduler()
                        songs = self.songs(count, ane)
                        executed = []
                        while self.sched.songs:
                            starts = self.sched.schedule()
                            self.assertTrue(starts)
                            executed.extend(starts)
                            for target, song in starts:
                                self.sched.songs.remove(song)
                                for slot in ('ane_synth', 'gpu_synth', 'remote_synth'):
                                    if getattr(self.sched, slot) is song:
                                        setattr(self.sched, slot, None)
                        self.assertEqual(sum(fn is w.run_remote for fn, _ in executed), count // 2)
                        self.assertEqual(len({id(song) for _, song in executed}), count)
                        self.assertEqual(len(executed), count)

    def test_remote_and_mac_start_together_without_local_pipeline(self):
        self.songs(2)
        with patch.object(w, 'CONCURRENT', False):
            self.assertEqual({fn for fn, _ in self.sched.schedule()}, {w.run_gpu, w.run_remote})

    def test_phone_reservation_survives_mac_finishing_first(self):
        songs = self.songs(4)
        starts = self.sched.schedule()
        self.assertIn((w.run_remote, songs[1]), starts)
        self.sched.gpu_synth = None
        self.sched.songs.remove(songs[0])
        self.assertEqual(self.sched.schedule(), [(w.run_gpu, songs[2])])
        self.assertEqual(songs[3].state, w.SYNTH_WAIT)

    def test_separate_submissions_share_wave_and_drained_wave_resets(self):
        first = self.songs(1)[0]
        second = self.songs(1)[0]
        self.assertFalse(first.remote_assigned)
        self.assertTrue(second.remote_assigned)
        self.sched.songs.clear()
        self.assertFalse(self.songs(1)[0].remote_assigned)

    def test_fallback_for_disconnect_failure_length_and_gpu_only(self):
        for reason in ('disconnect', 'failure', 'length', 'gpu_only'):
            with self.subTest(reason=reason):
                self.sched = w.Scheduler()
                songs = self.songs(2)
                if reason == 'failure': songs[1].remote_failed = True
                if reason == 'length': songs[1].codec = [0] * 5000
                if reason == 'gpu_only': songs[1].allow_ane = False
                with patch.object(w, 'REMOTE', None if reason == 'disconnect' else object()):
                    self.sched.schedule()
                self.assertFalse(songs[1].remote_assigned)
                self.sched.gpu_synth = None
                self.sched.songs.remove(songs[0])
                self.assertEqual(self.sched.schedule(), [(w.run_gpu, songs[1])])

    def test_length_fallback_reports_why_the_iphone_was_skipped(self):
        self.sched = w.Scheduler()
        songs = self.songs(2, ane=True, frames=5000)
        with patch.object(w, 'log') as logger, patch.object(w, 'emit') as emitter:
            self.sched.schedule()

        rows = w.remote_rows(songs[1])
        logger.assert_any_call(
            f"{songs[1].label} will use the Mac: {songs[1].frames} frames require a {rows}-row iPhone program; "
            f"current remote limit is {w.REMOTE_MAX_ROWS}"
        )
        details = [call.kwargs.get('detail', '') for call in emitter.call_args_list]
        self.assertIn(
            f"too long for iPhone: {rows}-row program exceeds {w.REMOTE_MAX_ROWS}-row limit · using Mac",
            details,
        )

    def test_cancelled_phone_song_never_starts(self):
        songs = self.songs(2)
        songs[1].cancel.set()
        self.assertEqual(self.sched.schedule(), [(w.run_gpu, songs[0])])

    def test_remote_overlaps_token_generation_and_rendering(self):
        for state, target in ((w.QUEUED, w.run_batch), (w.RENDER_WAIT, w.run_render)):
            with self.subTest(state=state), patch.object(w, 'CONCURRENT', False):
                self.sched = w.Scheduler()
                songs = self.songs(2)
                songs[0].state = state
                starts = self.sched.schedule()
                self.assertEqual({fn for fn, _ in starts}, {target, w.run_remote})

    def test_remote_prefill_independently_pauses_gpu(self):
        self.sched.remote_prefill = True
        self.assertTrue(self.sched.gpu_wanted_elsewhere())
        self.sched.remote_prefill = False
        self.assertFalse(self.sched.gpu_wanted_elsewhere())


if __name__ == '__main__':
    unittest.main()
