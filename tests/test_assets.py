import importlib.util
import json
from pathlib import Path
import pytest


def module(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_asset_import_is_physical_and_preserves_source(tmp_path):
    mod=module('import_assets')
    source=tmp_path/'main'; study=source/'outputs/study'; repo=tmp_path/'side'
    (source/'configs').mkdir(parents=True);(source/'models/qwen-math').mkdir(parents=True)
    (study/'data').mkdir(parents=True);repo.mkdir()
    (source/'configs/assets.json').write_text('{}')
    (source/'configs/container.sha256').write_text('abc')
    (study/'manifest.json').write_text('{}')
    (study/'data/questions.json').write_text(json.dumps({'questions':[]}))
    weight=tmp_path/'cached-weight';weight.write_bytes(b'weights')
    (source/'models/qwen-math/model.safetensors').symlink_to(weight)
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    mod.ROOT=repo
    mod.import_assets(source,study)
    copied=repo/'models/qwen-math/model.safetensors'
    assert copied.read_bytes()==b'weights' and not copied.is_symlink()
    assert {str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}==before
    manifest=json.loads((repo/'inputs/assets.json').read_text())
    assert all(mod.checksum(repo/p)==h for p,h in manifest['files'].items())
    with pytest.raises(ValueError,match='already imported'):mod.import_assets(source,study)


def test_import_rejects_source_checkout(tmp_path):
    mod=module('import_assets');mod.ROOT=tmp_path/'main'
    with pytest.raises(ValueError,match='outside'):mod.import_assets(mod.ROOT,mod.ROOT/'outputs/study')
