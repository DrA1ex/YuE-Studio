"""Synthetic integration checks for hummed/open scores in the advanced fork."""
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from test_upstream_04 import worker
from test_hum import HEADER

ABC = HEADER + 'V: Vocal\nz32|\nV: Ins\nC8D8E8F8|\n'

@pytest.mark.parametrize('kind,opening,instrumental', [('GENERATED', True, False), ('COVER', True, False), ('COVER', False, False), ('GENERATED', True, True)])
def test_opening_admission_preserves_fork_controls(tmp_path, kind, opening, instrumental):
    submitted = []
    with patch.object(worker, 'OUTPUT_DIR', tmp_path), patch.object(worker.SCHED, 'submit', submitted.extend), patch.object(worker, 'emit'):
        worker.submit_generate(dict(style='piano', lyrics='words', cot='melody', abc=ABC, abc_open=opening,
                                    instrumental=instrumental, kind=kind, target_seconds=10, max_tokens=3000,
                                    title='My tune', source_path='/source.wav', prompt_fidelity=1, style_fidelity=1))
    s = submitted[0]
    assert s.needs_plan is opening and s.request.abc_open is opening
    assert s.title == 'My tune' and s.source_path == '/source.wav' and s.kind == kind
    assert s.limit == (250 if kind == 'COVER' and not opening else 3000)
    assert s.top_p == pytest.approx(.85)
    if opening:
        assert 'V: Vocal\nC8D8E8F8|' in s.request.abc


def test_zero_source_fidelity_clears_open_score(tmp_path):
    submitted = []
    with patch.object(worker, 'OUTPUT_DIR', tmp_path), patch.object(worker.SCHED, 'submit', submitted.extend), patch.object(worker, 'emit'):
        worker.submit_generate(dict(style='piano', lyrics='words', cot='melody', abc=ABC, abc_open=True,
                                    kind='COVER', source_fidelity=0))
    assert submitted[0].request.abc is None
    assert not submitted[0].request.abc_open


def test_hum_analysis_uses_vocal_prompt_and_skips_lyrics_style(tmp_path):
    from scipy.io import wavfile
    import transcribe_cover
    audio = tmp_path / 'hum.wav'
    wavfile.write(audio, 16000, np.sin(np.arange(32000) * .1).astype(np.float32))
    model = MagicMock()
    transcribe = model.eval.return_value.to.return_value.transcribe
    transcribe.return_value = {'abc': ABC, 'warnings': []}
    for index in range(2):
        out = tmp_path / f'out{index}'
        with patch.object(sys, 'argv', ['transcribe_cover', str(audio), '--output', str(out), '--cache-dir', str(tmp_path/'cache'), '--hum']), \
             patch('transformers.AutoModel.from_pretrained', return_value=model) as load, \
             patch('transformers.pipeline') as optional, patch.object(transcribe_cover, 'describe_style') as style, \
             patch.object(transcribe_cover, 'event'):
            assert transcribe_cover.main() == 0
            optional.assert_not_called(); style.assert_not_called()
            if index: load.assert_not_called()
        data = json.loads((out/'cover_analysis.json').read_text())
        assert data['task'] == 'melody-vocal' and data['lyrics'] == ''
        assert (out/'score.abc').read_text().strip()
    assert transcribe.call_count == 1
    assert transcribe.call_args.kwargs['prompts'][-1] == 'melody_vocal'


@pytest.mark.parametrize('cached', [False, True])
def test_ane_compile_phase_only_for_uncached_program(cached):
    from types import SimpleNamespace
    from yue2 import nar
    from yue2.ane import runtime
    phases = []
    programs = SimpleNamespace(loaded={(512, 1024): []} if cached else {}, ensure=MagicMock(side_effect=RuntimeError('synthetic-stop')))
    model = SimpleNamespace(training=False, _yue2_ane_weights=object())
    chunk = SimpleNamespace(noise=[0] * 100, ar_tokens=[1] * 50)
    with patch('yue2.lean.is_lean', return_value=False), patch.object(runtime, 'weights_for'), \
         patch.object(runtime, 'programs_for', return_value=programs), \
         patch.object(runtime, 'buckets_for', return_value=(512, 1024)), patch.object(nar, 'song_chunks', return_value=[chunk]):
        with pytest.raises(RuntimeError, match='synthetic-stop'):
            nar.synthesize(model, [], [], 42, engine='ane', on_phase=phases.append)
    assert any('compiling the Neural Engine program' in p for p in phases) is (not cached)
    programs.ensure.assert_called_once()
