"""Synthetic 0.4 integration checks: no checkpoint, model download or iPhone required."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('worker04', ROOT / 'tools/yue2_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
from yue2.protocol import SongRequest
from yue2.remote.protocol import Connection, RemoteError


def song(tmp_path, frames=100, allow=True):
    value = worker.Song('run', 1, 42, SongRequest(style='piano', lyrics='hello', cot='off'),
                        tmp_path / 'song1', 'full', 32, 9000, allow_ane=allow)
    value.codec = [0] * frames
    value.state = worker.SYNTH_WAIT
    return value


@pytest.mark.parametrize('quality,engines,allowed', [
    ('draft', 'gpu', False), ('draft', 'gpu+ane', True),
    ('full', 'gpu', False), ('full', 'gpu+ane', True)])
def test_engine_selection_keeps_fork_metadata(tmp_path, quality, engines, allowed):
    submitted, events = [], []
    with patch.object(worker, 'OUTPUT_DIR', tmp_path), patch.object(worker.SCHED, 'submit', submitted.extend), \
         patch.object(worker, 'emit', lambda **event: events.append(event)):
        worker.submit_generate(dict(title='Night / Train', style='piano', lyrics='hello', cot='off',
                                    quality=quality, engines=engines, request_id='client-4', source_path='/source.wav',
                                    prompt_fidelity=1, style_fidelity=1))
    assert submitted[0].allow_ane is allowed
    assert submitted[0].temperature == pytest.approx(.85)
    assert submitted[0].top_p == pytest.approx(.85)
    started = next(e for e in events if e['event'] == 'started')
    assert started['request_id'] == 'client-4'
    assert started['songs'][0]['title'] == 'Night / Train'
    assert started['songs'][0]['source_path'] == '/source.wav'
    assert submitted[0].directory.parent.name.endswith('Night-Train')


def test_concurrent_runs_have_unique_directories(tmp_path):
    submitted = []
    with patch.object(worker, 'OUTPUT_DIR', tmp_path), patch.object(worker.SCHED, 'submit', submitted.extend), \
         patch.object(worker, 'emit'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: worker.submit_generate(dict(style='piano', lyrics='hi', cot='off', title='Same')), range(16)))
    assert len({s.directory for s in submitted}) == 16


@pytest.mark.parametrize('frames,allow,failed,expected', [
    (4094, True, False, True), (4095, True, False, False),
    (100, False, False, False), (100, True, True, False)])
def test_remote_capacity_and_engine_opt_out(tmp_path, frames, allow, failed, expected):
    s = song(tmp_path, frames, allow); s.remote_failed = failed
    with patch.object(worker, 'REMOTE', object()):
        assert worker.remote_can_take(s) is expected


def test_scheduler_uses_mac_and_phone_for_different_songs(tmp_path):
    a, b = song(tmp_path / 'a'), song(tmp_path / 'b')
    a.wants_ane = b.wants_ane = True
    scheduler = worker.Scheduler(); scheduler.songs = [a, b]
    with patch.object(worker, 'REMOTE', object()), patch.object(worker, 'CONCURRENT', True):
        starts = scheduler.schedule()
    assert starts == [(worker.run_ane, a), (worker.run_remote, b)]
    assert scheduler.ane_synth is a and scheduler.remote_synth is b


def test_scheduler_moves_running_gpu_song_to_phone(tmp_path):
    s = song(tmp_path); s.state = worker.SYNTHING
    scheduler = worker.Scheduler(); scheduler.songs = [s]; scheduler.gpu_synth = s
    with patch.object(worker, 'REMOTE', object()), patch.object(worker, 'CONCURRENT', True):
        assert scheduler.schedule() == []
    assert s.migrate == 'remote' and scheduler.remote_synth is s


def test_remote_failure_returns_song_to_mac(tmp_path):
    s = song(tmp_path); s.plan = SimpleNamespace(prefix=[])
    scheduler = worker.Scheduler(); scheduler.remote_synth = s
    client = SimpleNamespace(label=lambda: 'Test phone')
    with patch.object(worker, 'REMOTE', client), patch.object(worker, 'SCHED', scheduler), \
         patch.object(worker, 'acquire_model', return_value=(SimpleNamespace(offload_ar=False), object())), \
         patch.object(worker, 'synth_common', return_value=({}, 4, None)), \
         patch('yue2.nar.synthesize', side_effect=RemoteError('compiler refused')), \
         patch.object(worker, 'log'), patch.object(worker, 'emit'), patch.object(scheduler, 'tick') as tick:
        worker.run_remote(s)
    assert s.state == worker.SYNTH_WAIT and s.remote_failed
    assert scheduler.remote_synth is None
    tick.assert_called_once()


def test_render_accepts_missing_audio_and_keeps_cover_metadata(tmp_path):
    from yue2.pipeline import SymbolicPlan
    directory = tmp_path / 'song1'; directory.mkdir()
    np.save(directory / 'semantic.npy', np.array([1, 2, 3]))
    (directory / 'metadata.json').write_text(json.dumps(dict(title='Cover title', kind='COVER', source_path='/source.wav')))
    plan = SimpleNamespace(request=SongRequest(style='piano', lyrics='words', cot='off', seed=10))
    submitted, events = [], []
    with patch.object(SymbolicPlan, 'load', return_value=plan), patch.object(worker.SCHED, 'find', return_value=None), \
         patch.object(worker.SCHED, 'submit', submitted.extend), patch.object(worker, 'emit', lambda **e: events.append(e)):
        worker.submit_render(dict(path=str(directory / 'audio.flac'), engines='gpu', request_id='retry'))
    s = submitted[0]
    assert not s.allow_ane and not s.wants_ane
    assert s.title == 'Cover title' and s.kind == 'COVER' and s.source_path == '/source.wav'
    assert events[0]['request_id'] == 'retry'


def test_failed_render_emits_path_and_request_id(tmp_path):
    events = []
    with patch.object(worker, 'submit_render', side_effect=ValueError('bad tokens')), \
         patch.object(worker, 'emit', lambda **e: events.append(e)), patch.object(worker, 'log'), \
         patch.object(worker.traceback, 'print_exc'):
        worker.submit(dict(cmd='render', path='/missing/audio.flac', request_id='retry'))
    assert events[0]['event'] == 'failed' and events[0]['path'] == '/missing/audio.flac'
    assert events[-1]['event'] == 'error' and events[-1]['request_id'] == 'retry'


def connection(sock):
    conn = Connection.__new__(Connection); conn.sock = sock
    sock.settimeout(2)
    return conn


def test_wire_progress_and_binary_payload():
    left, right = socket.socketpair(); client, server = connection(left), connection(right)
    def reply():
        header, payload = server.recv()
        assert header['op'] == 'velocity' and payload == b'\x00\xff\x01'
        server.send(dict(op='progress', text='working'))
        server.send(dict(op='velocity', ok=True), b'\x80\x00')
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(reply)
        try:
            progress = []
            _, payload = client.call('velocity', b'\x00\xff\x01', on_progress=progress.append)
            future.result(timeout=3)
            assert progress == ['working'] and payload == b'\x80\x00'
        finally:
            client.close(); server.close()


@pytest.mark.parametrize('reply,match', [(dict(op='wrong', ok=True), 'expected a hello'),
                                        (dict(op='hello', ok=False, error='refused'), 'refused')])
def test_wire_rejects_wrong_or_failed_reply(reply, match):
    left, right = socket.socketpair(); client, server = connection(left), connection(right)
    try:
        server.send(reply)
        with pytest.raises(RemoteError, match=match): client.call('hello')
    finally:
        client.close(); server.close()


def test_wire_detects_truncated_frame():
    left, right = socket.socketpair(); client = connection(left)
    right.sendall(b'\x00\x00'); right.close()
    try:
        with pytest.raises(RemoteError, match='closed'): client.recv()
    finally: client.close()


def test_background_ane_compile_does_not_load_program():
    from yue2.ane.runtime import LayerPrograms
    programs = LayerPrograms.__new__(LayerPrograms)
    programs.lock = threading.RLock(); programs.loaded = {}
    with patch.object(programs, '_build', return_value=['program']) as build:
        programs.precompile(512, 1024)
        programs.precompile(512, 1024)
    build.assert_called_once_with(512, 1024, None, load=False)
    assert programs.loaded[(512, 1024)] == ['program']


def test_phone_disappears_before_synthesis_returns_to_mac(tmp_path):
    s = song(tmp_path); s.plan = SimpleNamespace(prefix=[])
    scheduler = worker.Scheduler(); scheduler.remote_synth = s
    with patch.object(worker, 'REMOTE', None), patch.object(worker, 'SCHED', scheduler), \
         patch.object(worker, 'acquire_model', return_value=(SimpleNamespace(offload_ar=False), object())), \
         patch.object(worker, 'synth_common', return_value=({}, 4, None)), patch.object(worker, 'clear_remote'), \
         patch.object(worker, 'log'), patch.object(worker, 'emit'), patch.object(scheduler, 'tick') as tick:
        worker.run_remote(s)
    assert s.state == worker.SYNTH_WAIT and scheduler.remote_synth is None
    tick.assert_called_once()


def test_phone_refusal_after_gpu_migration_requeues_locally(tmp_path):
    s = song(tmp_path); s.plan = SimpleNamespace(prefix=[]); s.migrate = 'remote'
    scheduler = worker.Scheduler(); scheduler.gpu_synth = scheduler.remote_synth = s
    with patch.object(worker, 'REMOTE', object()), patch.object(worker, 'SCHED', scheduler), \
         patch.object(worker, 'acquire_model', return_value=(SimpleNamespace(offload_ar=False), object())), \
         patch.object(worker, 'synth_common', return_value=({}, 4, None)), \
         patch('yue2.nar_switch.synthesize_switchable', side_effect=RemoteError('compiler refused')), \
         patch.object(worker, 'log'), patch.object(worker, 'emit'), patch.object(scheduler, 'tick'):
        worker.run_gpu(s)
    assert s.state == worker.SYNTH_WAIT and s.remote_failed and not s.migrate
    assert scheduler.remote_synth is None and scheduler.gpu_synth is None
    with patch.object(worker, 'REMOTE', object()):
        assert not worker.remote_can_take(s)
