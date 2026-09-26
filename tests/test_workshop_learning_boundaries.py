"""Execute notebook evaluation boundaries and real BYOD acquisition without models."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / 'tutorials/DIMER_MultiModel_Image_Classification_Workshop.ipynb'
)


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][index]['source'])


def test_baseline_does_not_access_test_until_after_freeze():
    class DevelopmentSplits(dict):
        def __getitem__(self, key):
            if key == 'test':
                raise AssertionError('test accessed before freeze')
            return super().__getitem__(key)

    ns = {'np': np, 'class_names': ['a', 'b'], 'splits': DevelopmentSplits(
        train=[{'label': x} for x in ['a', 'a', 'b']],
        validation=[{'label': x} for x in ['a', 'b']],
    )}
    exec(cell(16), ns)
    assert ns['majority_label'] == 'a'
    assert ns['majority_validation_metrics']['accuracy'] == 0.5
    assert 'majority_metrics' not in ns
    ns['splits'] = {'test': [{'label': 'b'}, {'label': 'b'}]}
    # Execute the actual post-freeze reference block, without foundation models.
    exec(cell(32).split('# Score the fixed training-derived reference')[1].partition('\n')[2], ns)
    assert ns['majority_metrics']['accuracy'] == 0.0
    assert ns['majority_validation_metrics']['accuracy'] == 0.5


def fixture(root, malformed=None):
    root.mkdir()
    rows = []
    for i in range(16):
        Image.new('RGB', (8, 8), (i, i * 3, i * 7)).save(root / f'{i}.png')
        rows.append({'id': str(i), 'file': f'{i}.png', 'label': 'a' if i < 8 else 'b'})
    frame = pd.DataFrame(rows)
    if malformed == 'missing_label':
        frame = frame.drop(columns='label')
    elif malformed == 'duplicate_id':
        frame.loc[1, 'id'] = '0'
    elif malformed == 'missing_image':
        frame.loc[0, 'file'] = 'absent.png'
    elif malformed == 'too_few':
        frame = frame.iloc[1:]
    frame.to_csv(root / 'labels.csv', index=False)
    return root


def acquire(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    ns = {}
    exec(cell(4), ns)
    ns.update(USE_BYOD=True, BYOD_PATH=str(path))
    exec(cell(11), ns)
    exec(cell(12), ns)
    return ns


@pytest.mark.parametrize('zipped', [False, True])
def test_real_byod_path_validates_splits_and_exports_actual_manifest(tmp_path, monkeypatch, zipped):
    path = fixture(tmp_path / 'input')
    if zipped:
        archive_path = tmp_path / 'input.zip'
        with zipfile.ZipFile(archive_path, 'w') as archive:
            for item in path.iterdir():
                archive.write(item, item.name)
        path = archive_path
    ns = acquire(tmp_path, monkeypatch, path)
    assert ns['dataset_source'] == 'BYOD'
    assert {key: len(value) for key, value in ns['splits'].items()} == {
        'train': 10, 'validation': 2, 'test': 4,
    }
    manifest = json.loads((tmp_path / 'outputs/data/dataset_manifest.json').read_text())
    assert manifest['class_names'] == ['a', 'b']
    assert manifest['duplicate_decoded_pixels_across_splits'] == 0


@pytest.mark.parametrize(('malformed', 'message'), [
    ('missing_label', 'must contain columns'),
    ('duplicate_id', 'ids must be unique'),
    ('missing_image', 'missing BYOD image'),
    ('too_few', 'at least 8 images'),
])
def test_real_byod_rejects_invalid_input_before_models(tmp_path, monkeypatch, malformed, message):
    path = fixture(tmp_path / 'input', malformed)
    with pytest.raises(ValueError, match=message):
        acquire(tmp_path, monkeypatch, path)
    assert not (tmp_path / 'outputs/data/dataset_manifest.json').exists()


def test_export_reports_actual_byod_classes_and_test_size(tmp_path, monkeypatch):
    ns = acquire(tmp_path, monkeypatch, fixture(tmp_path / 'input'))
    # Model results are stand-ins only for executing the export cell's data contract.
    ns.update(SELECTED_MODELS=[], CORE_MODELS=[], MODEL_SPECS={},
              majority_label='a', majority_metrics={'accuracy': 0.5, 'macro_f1': 1/3},
              test_table=pd.DataFrame(), core_test_table=pd.DataFrame())
    exec(cell(47), ns)
    manifest = json.loads((tmp_path / 'outputs/provenance/experiment_manifest.json').read_text())
    assert manifest['adaptation']['n_classes'] == 2
    assert 'six-class' not in manifest['adaptation']['method']
    assert '4-image held-out' in manifest['evidence_scope']
    assert Path(ns['bundle']).is_file()
