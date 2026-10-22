const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const sourcePath = path.join(__dirname, 'cleanup_observer.js');
const source = fs.existsSync(sourcePath) ? fs.readFileSync(sourcePath, 'utf8') : '(function(){ throw Error("cleanup observer not implemented"); })';

function fixture() {
  let clock = 0;
  function events(object) {
    const listeners = new Map();
    return Object.assign(object, {
      addEventListener(type, listener) { const rows = listeners.get(type) || []; rows.push(listener); listeners.set(type, rows); },
      removeEventListener(type, listener) { listeners.set(type, (listeners.get(type) || []).filter(row => row !== listener)); },
      dispose() { for (const listener of listeners.get('dispose') || []) listener({ type: 'dispose', target: this }); },
      listenerCount() { return (listeners.get('dispose') || []).length; }
    });
  }
  const canvas = { tagName: 'CANVAS' }, input = { contains: object => object === canvas };
  const camera = {}, controls = { object: camera, domElement: input, addEventListener() {} };
  const validBuffers = new Set(), attributeRows = new WeakMap(), materialRows = new WeakMap();
  const attributes = { get(attribute) { return attributeRows.get(attribute.isInterleavedBufferAttribute ? attribute.data : attribute); } };
  const gl = { domElement: canvas, getContext: () => ({ isBuffer: buffer => validBuffers.has(buffer) }),
    properties: { has: object => materialRows.has(object) }, info: { memory: { geometries: 8, textures: 4 }, programs: [], render: { calls: 1 } } };
  const scene = { uuid: 'viewer', children: [] };
  const state = { gl, scene, camera, controls, events: { connected: input } };
  const store = { getState: () => state };
  function CanvasCapture() {} function SceneMeshModel() {} function Viewer3D() {} function Provider() {}
  const provider = { type: Provider, memoizedProps: { store } };
  const capture = { type: CanvasCapture, memoizedProps: { domRef: { current: canvas } } };
  const primitive = { type: 'primitive', memoizedProps: { object: controls }, stateNode: controls };
  const defaults = { type: {}, memoizedProps: { makeDefault: true }, child: primitive };
  const viewer = { type: Viewer3D, memoizedProps: {}, memoizedState: { memoizedState: null, next: null } };
  provider.sibling = capture; capture.sibling = defaults; defaults.sibling = viewer;
  const root = { current: { child: provider } }, roots = new Set([root]);
  const cache = [], Loader = function GLTFLoader() {};
  const makeModel = label => {
    const position = {}, interleaved = { data: {}, isInterleavedBufferAttribute: true }, index = {};
    const geometry = events({ uuid: label + '-geometry', attributes: { position, color: interleaved }, index, boundsTree: {} });
    const material = events({ uuid: label + '-material' }); materialRows.set(material, {});
    const mesh = { uuid: label + '-mesh', isMesh: true, geometry, material, userData: { originalMaterial: material }, children: [] };
    const model = { uuid: label + '-scene', parent: scene, children: [mesh], __r3f: { root: store } };
    mesh.parent = model;
    const buffers = [position, interleaved.data, index].map(attribute => { const buffer = {}; validBuffers.add(buffer); attributeRows.set(attribute, { buffer }); return buffer; });
    const fiber = { type: SceneMeshModel, memoizedProps: { scene: model, url: label + '.glb' } };
    return { model, mesh, geometry, material, position, interleaved, index, buffers, fiber };
  };
  const activate = row => { scene.children = row ? [row.model] : []; viewer.sibling = row?.fiber || null; };
  const install = (WeakRefType = WeakRef) => vm.runInNewContext('(' + source + ')(roots,cache,Loader,attributes)', { roots, cache, Loader, attributes, WeakRef: WeakRefType, performance: { now: () => ++clock } });
  return { install, makeModel, activate, root, roots, viewer, scene, store, cache, Loader, attributes, attributeRows, materialRows, validBuffers, gl };
}

test('authenticates the empty main viewer and rejects a foreign model store', () => {
  const f = fixture(), api = f.install();
  assert.equal(api.snapshot().currentModels.length, 0);
  api.stop();
  const foreign = fixture(), row = foreign.makeModel('bad'); row.model.__r3f.root = {};
  foreign.activate(row);
  assert.throws(foreign.install, /authenticated main viewer/);
});

test('calibrates real tracked buffers/materials and filters the exact loader cache key', () => {
  const f = fixture(), api = f.install(), row = f.makeModel('master'); f.activate(row);
  f.cache.push({ keys: [f.Loader, 'master.glb'], response: [{ scene: row.model }] },
    { keys: [function ForeignLoader() {}, 'master.glb'], response: [{ scene: row.model }] },
    { keys: [f.Loader, 'other.glb'], response: [{ scene: row.model }] });
  const snapshot = api.trackModel('master');
  assert.equal(snapshot.models[0].cache.count, 1);
  assert.deepEqual(Array.from(snapshot.models[0].cache.sceneUuids), ['master-scene']);
  assert.equal(snapshot.resources.find(resource => resource.kind === 'material').loadedPropertyPresent, true);
  assert.equal(snapshot.attributes.length, 3);
  assert.ok(snapshot.attributes.every(attribute => attribute.loadedRecordPresent && attribute.loadedBufferValid));
  assert.throws(() => api.trackModel('master'), /already tracked/);
  api.stop();
});

test('observes disposal, BVH clearing, cache removal and exact buffer deletion after detach', () => {
  const f = fixture(), api = f.install(), row = f.makeModel('master'); f.activate(row);
  f.cache.push({ keys: [f.Loader, 'master.glb'], response: [{ scene: row.model }] }); api.trackModel('master');
  row.geometry.dispose(); row.material.dispose(); row.geometry.boundsTree = null; row.model.parent = null;
  for (const attribute of [row.position, row.interleaved.data, row.index]) f.attributeRows.delete(attribute);
  for (const buffer of row.buffers) f.validBuffers.delete(buffer);
  f.materialRows.delete(row.material); f.cache.splice(0, 1); f.activate(null);
  const snapshot = api.snapshot();
  assert.equal(snapshot.models[0].attachedToMainScene, false);
  assert.equal(snapshot.models[0].cache.count, 0);
  assert.equal(snapshot.models[0].activeSceneMatches, 0);
  assert.equal(snapshot.resources.find(resource => resource.kind === 'geometry').bvhPresent, false);
  assert.ok(snapshot.resources.every(resource => resource.disposals === 1));
  assert.equal(snapshot.resources.find(resource => resource.kind === 'material').propertyPresent, false);
  assert.ok(snapshot.attributes.every(attribute => attribute.recordPresent === false && attribute.bufferValid === false));
  assert.equal(snapshot.census.current.length, 0);
  api.stop();
});

test('separates current hook/reference ownership from alternate-only old model props', () => {
  const f = fixture(), api = f.install(), row = f.makeModel('master'); f.activate(row); api.trackModel('master');
  f.activate(null); row.model.parent = null;
  f.root.current.alternate = { child: row.fiber };
  f.viewer.memoizedState = { memoizedState: { current: row.model }, next: { memoizedState: [() => {}, [row.model]], next: null } };
  const snapshot = api.snapshot();
  assert.ok(snapshot.census.current.some(match => match.slot === 'hook.ref.current' && match.label === 'master'));
  assert.ok(snapshot.census.current.some(match => match.slot === 'hook.memo.dependencies' && match.label === 'master'));
  assert.ok(snapshot.census.alternate.some(match => match.slot === 'props.scene' && match.label === 'master'));
  assert.equal(snapshot.census.complete, true);
  api.stop();
});

test('marks a malformed cyclic fiber census incomplete instead of looping or claiming absence', () => {
  const f = fixture(), api = f.install(); f.viewer.sibling = f.viewer;
  assert.equal(api.snapshot().census.complete, false);
  api.stop();
});

test('stop removes owned listeners, releases handles and returns an idempotent scalar receipt', () => {
  const f = fixture(), api = f.install(), row = f.makeModel('master'); f.activate(row); api.trackModel('master');
  assert.equal(row.geometry.listenerCount(), 1); assert.equal(row.material.listenerCount(), 1);
  const receipt = api.stop();
  assert.equal(row.geometry.listenerCount(), 0); assert.equal(row.material.listenerCount(), 0);
  assert.equal(receipt.listenersRemoved, 2); assert.equal(receipt.rootHandlesReleased, 1);
  assert.equal(receipt.cacheReleased, true); assert.equal(receipt.attributesReleased, true);
  assert.equal(JSON.stringify(api.stop()), JSON.stringify(receipt));
  assert.throws(() => api.snapshot(), /stopped/);
  assert.equal(typeof JSON.parse(JSON.stringify(receipt)).stopped, 'boolean');
});

test('collected resource weak handles do not falsely match an empty scene node without geometry', () => {
  let collected = false;
  class SimulatedWeakRef {
    constructor(object) { this.ref = new WeakRef(object); }
    deref() { const object = this.ref.deref(); return collected && /-geometry$|-material$/.test(object?.uuid || '') ? undefined : object; }
  }
  const f = fixture(), api = f.install(SimulatedWeakRef), row = f.makeModel('master'); f.activate(row); api.trackModel('master');
  f.activate(null); row.model.parent = null; collected = true;
  assert.equal(api.snapshot().models[0].activeSceneMatches, 0);
  api.stop();
});

test('accepts a weak root registry and reports current app state without exposing model URLs', () => {
  const f = fixture(); f.roots.clear(); f.roots.add(new WeakRef(f.root));
  const api = f.install(), row = f.makeModel('master'); f.activate(row); api.trackModel('master');
  f.viewer.memoizedState = { memoizedState: { currentJob: { id: 'import-1', status: 'done', outputUrl: 'master.glb' },
    apiUrl: '', setCurrentJob() {}, meshSelected: true, meshHistory: ['master.glb'] }, next: null };
  const state = api.snapshot().census.appState;
  assert.equal(state.available, true); assert.equal(state.meshSelected, true); assert.equal(state.historyCount, 1);
  assert.deepEqual(Array.from(state.currentJob.modelLabels), ['master']);
  assert.equal(JSON.stringify(state).includes('master.glb'), false);
  api.stop();
});
