"""One immutable authenticated brick run; strings/counts never grant support."""
from dataclasses import dataclass
import hashlib
import json
from ._source import load_pinned

NAME = 'brick-1957891-v1'
CONTRACT_SHA256 = '82bf47920f667a3a445d5e70297a0057fae78a073d7ae9c903f0827319a8dfdd'
TEXTURE_SHA256 = '4c74343c961e74bf3cfde4c24bd2a21b23063e17f0468b7f3049311fa9dc4289'
GEOMETRY_SHA256 = 'd4de05fc82315e44a739c9815a07a531555e108af05c619ea0426d0c630ef861'
INITIAL_COORDINATES_SHA256 = 'e350ce386f21a9ee8473fda12a8839bf8c37ae0c1375c34c56cea88f81859520'
FINAL_COORDINATES_SHA256 = '690670fe58a7021c51fef9fad6457ad8e66715d13a88239adabdffa8a05d9976'
TEXTURE_METADATA_SHA256 = '25adb1a50b8b31900ebc56d87f45b7fe23dee9305d97b458f70a60799d59e2a0'
GEOMETRY_METADATA_SHA256 = 'a42340cef604134b2866a3556f6d74c5f631073eb908fb60a64d90139fac6407'
CHECKPOINT_SHA256 = '97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c'
CONFIG_SHA256 = '22074e4ab5b28e2b72d33a3fe61bf416134b89e12d0fc5c729f37da75da0c4d1'
HEADER_SHA256 = 'ad3ed458569d937eace6ac42d9b6aa83853d654ebd0847791bbf19e2e31ab02b'
PARENT_ROWS = (4430, 21391, 97313, 438684)
CHILD_ROWS = (21391, 97313, 438684, 1957891)
PINS = {
    'experimental/extensions/trellis2-amd/inference_decoder/run_contract.py': '0998c6e0c9a8d5c262b6f258f60fb11d82e8b51aec2c5fd48c771c33ef999f57',
    'experimental/extensions/trellis2-amd/inference_decoder/artifact_io.py': '10a2792b47ab24ec64cf5d911c37fa12ac789a1d7134c5221d0479ced1bdb989',
    'scripts/compatibility/amd_texture_decoder_inputs.py': '87bd6437412df561100b174e945c2fa040a9c02ee1ac2ea8c70a325b8f1a3d85',
    'scripts/compatibility/amd_texture_sampling_artifact.py': '5eaf35c08cf2e96f46dfe8bb32fa835c079ef34ce5a322a0afd411ee7cf9474c',
    'scripts/compatibility/amd_texture_field_artifact.py': '77d9592662653413e9b40f81741e47e2a23739e9e5d013d48b6cefff1831d3b5',
    'experimental/extensions/trellis2-amd/texture_decoder_loader.py': '409bf691bb3d7f3638d40cb46f02250111249af31f19a4377345da84ec2fb988',
}
_CONTRACT_SOURCE = 'experimental/extensions/trellis2-amd/inference_decoder/run_contract.py'
contracts = load_pinned('profile_contract', _CONTRACT_SOURCE, PINS[_CONTRACT_SOURCE])


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class Profile:
    name: str
    _geometry_json: bytes
    initial_rows: int = 4430
    parent_rows: tuple = PARENT_ROWS
    child_rows: tuple = CHILD_ROWS
    final_rows: int = 1957891
    run_contract_sha256: str = CONTRACT_SHA256

    def document(self):
        return dict(schema='modly.amd.brick-guided-pbr-profile.v1', name=self.name,
                    initial_rows=self.initial_rows, parent_rows=list(self.parent_rows),
                    child_rows=list(self.child_rows), final_rows=self.final_rows,
                    run_contract_sha256=self.run_contract_sha256,
                    geometry_contract=json.loads(self._geometry_json),
                    texture_sha256=TEXTURE_SHA256, geometry_sha256=GEOMETRY_SHA256,
                    texture_sidecar_sha256=TEXTURE_METADATA_SHA256, geometry_sidecar_sha256=GEOMETRY_METADATA_SHA256,
                    initial_coordinates_sha256=INITIAL_COORDINATES_SHA256,
                    final_coordinates_sha256=FINAL_COORDINATES_SHA256,
                    checkpoint_sha256=CHECKPOINT_SHA256, config_sha256=CONFIG_SHA256,
                    checkpoint_header_sha256=HEADER_SHA256, source_pins=dict(PINS))

    @property
    def sha256(self):
        return hashlib.sha256(_canonical(self.document())).hexdigest()


def authenticate(name, geometry_document):
    if type(name) is not str or name != NAME:
        raise ValueError('Exact explicit brick-1957891-v1 opt-in required')
    contract = contracts.parse(geometry_document)
    if contract.sha256 != CONTRACT_SHA256 or contract.final_support_profile != name:
        raise ValueError('Exact authenticated brick geometry run contract required')
    return Profile(name, _canonical(contract.document()))


def require_authenticated(profile):
    if type(profile) is not Profile:
        raise ValueError('Authenticated brick Profile object required')
    try:
        expected = authenticate(profile.name, json.loads(profile._geometry_json))
        if profile != expected or profile.document() != expected.document():
            raise ValueError('Authenticated brick profile fields differ')
    except (TypeError, AttributeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed authenticated brick profile') from exc
    return profile
