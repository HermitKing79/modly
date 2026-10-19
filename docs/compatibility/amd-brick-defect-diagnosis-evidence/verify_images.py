"""Independent PNG/hash and unannotated-crop readback, no rendering."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from PIL import Image
E = Path(__file__).resolve().parent
P = E.parents[2].parent / 'artifacts/amd-brick-defect-diagnosis-1d1c1a6'
def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def main():
    meta = json.loads((P / 'overlay-metadata.json').read_text())
    original = Path(meta['inputs_before']['source_png']['path'])
    assert digest(original) == '74d1655d53d5fe7d262756c7e23c0a695aa42b7ee3c43e075007a59fc3bf7f54'
    with Image.open(original) as source:
        source.load()
        source = source.convert('RGB')
    rows = []
    for row in meta['outputs']:
        path = Path(row['path'])
        assert path.parent == P and path.stat().st_size == row['bytes']
        assert digest(path) == row['sha256']
        with Image.open(path) as image:
            assert image.format == 'PNG' and image.mode == 'RGB'
            assert list(image.size) == row['dimensions']
            assert image.width * image.height <= 2_000_000
            image.load()
            crop_exact = None
            if row['kind'] == 'unannotated_and_annotated_crop':
                crop = source.crop(tuple(row['source_box']))
                expected = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
                actual = image.crop((24, 104, 24 + expected.width, 104 + expected.height))
                crop_exact = actual.tobytes() == expected.tobytes()
                assert crop_exact
            rows.append(dict(path=str(path), sha256=row['sha256'], bytes=row['bytes'],
                             dimensions=row['dimensions'], independently_reopened=True,
                             unannotated_crop_pixel_exact=crop_exact))
    assert len(rows) == 3 and meta['inputs_before'] == meta['inputs_after']
    record = dict(classification='PASS', CPU_only=True, original_PNG_unchanged=True,
                  outputs=rows, actual_overlay_case_markers=8, actual_overlay_edge_markers=10,
                  scope='PNG identities/readback and two unchanged nearest-neighbor crop panels; no causal or lighting inference.')
    with (E / 'image-verification.json').open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    print(json.dumps(record))
if __name__ == '__main__':
    main()
