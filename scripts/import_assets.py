"""Copy existing Duke assets into this repository, without changing the source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def import_assets(project, study):
    project, study = project.resolve(), study.resolve()
    if project == ROOT or ROOT in project.parents or project in ROOT.parents:
        raise ValueError('Copy this side study outside the main project before importing assets')
    destination = ROOT / 'inputs/assets.json'
    if destination.exists():
        raise ValueError('Assets already imported; preserve this snapshot or use a new repository copy')
    questions = study / 'data/questions.json'
    parsed = json.loads(questions.read_text())
    if not isinstance(parsed.get('questions'), list):
        raise ValueError('Expected a prepared online study question snapshot')
    files = [(questions, ROOT / 'inputs/questions.json')]
    for name in ('assets.json', 'container.sha256'):
        files.append((project / 'configs' / name, ROOT / 'configs' / name))
    model = project / 'models/qwen-math'
    files.extend((p, ROOT / 'models/qwen-math' / p.relative_to(model))
                 for p in sorted(model.rglob('*')) if p.is_file())
    if not any(src.name.endswith('.safetensors') for src, _ in files):
        raise ValueError('Missing model weights in source project')
    for src, dest in files:
        if not src.is_file():
            raise FileNotFoundError(src)
        if dest.is_symlink():
            raise ValueError(f'Destination must be a physical copy: {dest}')
    hashes = {}
    for src, dest in files:
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = checksum(src)
        if dest.exists() and checksum(dest) != expected:
            raise ValueError(f'Different destination already exists: {dest}')
        if not dest.exists():
            temp = dest.with_name(dest.name + '.copying')
            shutil.copyfile(src, temp)  # Dereference Hugging Face cache symlinks.
            if checksum(temp) != expected:
                raise ValueError(f'Copy checksum mismatch: {src}')
            temp.replace(dest)
        hashes[str(dest.relative_to(ROOT))] = expected
    destination.write_text(json.dumps(dict(files=hashes,
        original_study_manifest_sha256=checksum(study/'manifest.json'),
        scope='Physical read-only-source copies; source questions were previously inspected.'), indent=2)+'\n')
    print(f'Copied and verified {len(files)} files. No source files modified.')
    print('Model/data files are ignored by Git; transfer them separately when moving machines.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from-project', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    a = p.parse_args()
    study = a.study if a.study.is_absolute() else a.from_project/a.study
    import_assets(a.from_project, study)


if __name__ == '__main__':
    main()
