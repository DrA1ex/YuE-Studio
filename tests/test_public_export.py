"""Public export rejects directory residue and retains usable HF models."""
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAEConfig
from yue2.storage import copy_model_files, resolve_model, verify_model_snapshot
from yue2.models import MAIN_MODEL_ID, MAIN_MODEL_REVISION, VAE_MODEL_ID, VAE_MODEL_REVISION
from test_model import tiny_config


@pytest.mark.parametrize('config_class', [YuE2Config, YuE2VAEConfig])
def test_unknown_metadata_is_not_loaded_or_reserialized(config_class):
    config = config_class(unused_option={'private_note': 'DO_NOT_EXPORT'},
                          _name_or_path='/private/build/source')
    config.extra_note = 'DO_NOT_EXPORT'
    text = json.dumps(config.to_dict())
    assert 'DO_NOT_EXPORT' not in text and '/private/build' not in text
    assert not hasattr(config, 'unused_option')


def test_model_export_omits_unlisted_code_reports_and_cache(tmp_path):
    torch.set_num_threads(1)
    source, exported = tmp_path / 'source', tmp_path / 'export'
    model = YuE2ForCausalLM(tiny_config()).eval()
    model.save_pretrained(source)
    (source / 'notes.json').write_text('{"private_note":"DO_NOT_EXPORT"}')
    (source / 'extra_model.py').write_text('DO_NOT_EXPORT')
    (source / 'modeling_yue2.py').write_text('DO_NOT_EXPORT')
    (source / '.cache').mkdir()
    (source / '.cache/report.txt').write_text('DO_NOT_EXPORT')
    config = json.loads((source / 'config.json').read_text())
    config['unused_option'] = 'DO_NOT_EXPORT'
    (source / 'config.json').write_text(json.dumps(config))
    copy_model_files(source, exported)
    assert not (exported / 'notes.json').exists()
    assert not (exported / 'extra_model.py').exists()
    assert not (exported / '.cache').exists()
    assert 'DO_NOT_EXPORT' not in (exported / 'config.json').read_text()
    assert 'DO_NOT_EXPORT' not in (exported / 'modeling_yue2.py').read_text()
    restored = AutoModelForCausalLM.from_pretrained(exported, trust_remote_code=True).eval()
    ids = torch.tensor([[1, 5, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


def test_export_refuses_nonempty_destination(tmp_path):
    destination = tmp_path / 'output'; destination.mkdir()
    (destination / 'old_report.txt').write_text('DO_NOT_EXPORT')
    with pytest.raises(FileExistsError):
        copy_model_files(tmp_path / 'source', destination)


def test_pipeline_export_refuses_residue_at_the_root(tmp_path):
    from yue2.pipeline import YuE2Pipeline
    pipe = YuE2Pipeline.__new__(YuE2Pipeline)
    (tmp_path / 'old_report.txt').write_text('DO_NOT_EXPORT')
    with pytest.raises(FileExistsError, match='empty pipeline'):
        pipe.save_pretrained(tmp_path)


def test_hf_download_is_restricted_to_inference_files(monkeypatch, tmp_path):
    import huggingface_hub
    captured = {}
    def download(*args, **kwargs):
        captured.update(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    resolve_model('example/YuE2', revision='a' * 40, local_files_only=True)
    assert '*.json' not in captured['allow_patterns'] and '*.py' not in captured['allow_patterns']
    assert 'config.json' in captured['allow_patterns']
    assert captured['revision'] == 'a' * 40 and captured['local_files_only']


def test_default_pipeline_pins_first_party_model_revisions(monkeypatch, tmp_path):
    from yue2.pipeline import YuE2Pipeline
    calls = []

    def resolve(model, revision=None, **kwargs):
        calls.append((str(model), revision, kwargs.get("local_files_only")))
        return tmp_path

    class ProbePipeline(YuE2Pipeline):
        def __init__(self, model_dir, vae_dir, **kwargs):
            self.load_timing = {}

    monkeypatch.setattr("yue2.pipeline.resolve_model", resolve)
    ProbePipeline.from_pretrained(local_files_only=True, progress=False)
    assert calls == [(MAIN_MODEL_ID, MAIN_MODEL_REVISION, True),
                     (VAE_MODEL_ID, VAE_MODEL_REVISION, True)]


def test_resolve_model_forwards_repair_download_options(monkeypatch, tmp_path):
    import huggingface_hub
    captured = {}

    def download(*args, **kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    resolve_model("example/YuE2", force_download=True, max_workers=3)
    assert captured["force_download"] is True and captured["max_workers"] == 3



def test_verify_model_snapshot_rejects_missing_indexed_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model-00001-of-00002.safetensors",
                       "b": "model-00002-of-00002.safetensors"}
    }))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"partial")
    with pytest.raises(FileNotFoundError, match="Missing model shard"):
        verify_model_snapshot(tmp_path)


def test_verify_model_snapshot_accepts_complete_indexed_shards(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model-00001-of-00002.safetensors",
                       "b": "model-00002-of-00002.safetensors"}
    }))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"a")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"b")
    assert verify_model_snapshot(tmp_path) == tmp_path
