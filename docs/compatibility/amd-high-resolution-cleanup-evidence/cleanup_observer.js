(function (roots, cache, Loader, attributes) {
  'use strict';
  // Temporary ownership checks only. No scene/cache mutation, forced GC, render
  // wrapper, frame subscription, input listener, imports or retained fiber graph.
  if (!roots || !Array.isArray(cache) || typeof Loader !== 'function' || typeof attributes?.get !== 'function')
    throw Error('Authenticated roots, loader cache, loader and attributes required');
  let rootRefs = Array.from(roots, entry => typeof entry?.deref === 'function' ? entry : new WeakRef(entry));
  roots = null;
  if (rootRefs.length > 32) throw Error('Root census cap');
  const LIMIT = 12000;
  let stopped = false, receipt = null, storeRef, glRef;
  let models = [], resources = [], attributeRecords = [], seen = new WeakMap();
  function walk(start) {
    const rows = [], stack = start ? [start] : [], visited = new Set();
    let complete = true;
    while (stack.length) {
      const fiber = stack.pop();
      if (visited.has(fiber)) { complete = false; continue; }
      if (rows.length >= LIMIT) { complete = false; break; }
      visited.add(fiber); rows.push(fiber);
      if (fiber.sibling) stack.push(fiber.sibling);
      if (fiber.child) stack.push(fiber.child);
    }
    return { rows, complete };
  }
  const name = fiber => fiber.type && (fiber.type.name || fiber.type.displayName) || String(fiber.type || 'unknown');
  function discover() {
    const matches = [];
    let complete = true;
    for (const ref of rootRefs) {
      const root = ref.deref(); if (!root) continue;
      const tree = walk(root.current); complete = complete && tree.complete;
      const fibers = tree.rows;
      for (const capture of fibers.filter(fiber => name(fiber) === 'CanvasCapture')) {
        const canvas = capture.memoizedProps?.domRef?.current;
        if (!canvas || canvas.tagName !== 'CANVAS') continue;
        for (const provider of fibers) {
          const store = provider.memoizedProps?.store;
          if (typeof store?.getState !== 'function') continue;
          const state = store.getState(), controls = state.controls;
          if (state.gl?.domElement !== canvas || !state.scene || !state.camera || !controls ||
              controls.object !== state.camera || typeof controls.addEventListener !== 'function') continue;
          const input = state.events?.connected || canvas;
          if (controls.domElement !== input || !(input === canvas || input.contains?.(canvas))) continue;
          const defaults = fibers.some(fiber => fiber.memoizedProps?.makeDefault === true &&
            walk(fiber.child).rows.some(child => child.memoizedProps?.object === controls && child.stateNode === controls));
          if (!defaults) continue;
          const currentModels = fibers.filter(fiber => name(fiber) === 'SceneMeshModel');
          if (currentModels.some(fiber => fiber.memoizedProps?.scene?.__r3f?.root !== store)) continue;
          if (!matches.some(match => match.store === store)) matches.push({ store, currentModels });
        }
      }
    }
    if (matches.length !== 1) throw Error('Expected one authenticated main viewer; got ' + matches.length);
    return { ...matches[0], complete };
  }
  // This block's strong discovery values do not escape into the API closures.
  {
    const found = discover();
    if (!found.complete) throw Error('Initial viewer census incomplete');
    storeRef = new WeakRef(found.store); glRef = new WeakRef(found.store.getState().gl);
  }
  const live = () => { if (stopped) throw Error('Cleanup observer stopped'); };
  function listenerFor(row) { return () => { row.disposals++; row.lastDisposeAt = performance.now(); }; }
  function registerResource(object, kind, label) {
    if (!object) return;
    const prior = seen.get(object);
    if (prior) { if (!prior.labels.includes(label)) prior.labels.push(label); return; }
    const gl = glRef.deref();
    const row = { kind, label, labels: [label], uuid: object.uuid, ref: new WeakRef(object), disposals: 0, lastDisposeAt: null,
      loadedPropertyPresent: kind === 'material' && typeof gl?.properties?.has === 'function' ? gl.properties.has(object) : null };
    row.listener = listenerFor(row); object.addEventListener('dispose', row.listener);
    seen.set(object, row); resources.push(row);
  }
  function trackAttribute(attribute, geometryUuid, attributeName, label) {
    if (!attribute) return;
    const record = attributes.get(attribute), buffer = record?.buffer;
    const context = glRef.deref()?.getContext();
    attributeRecords.push({ label, geometryUuid, name: attributeName, ref: new WeakRef(attribute),
      bufferRef: buffer ? new WeakRef(buffer) : null, loadedRecordPresent: !!record,
      loadedBufferValid: buffer && typeof context?.isBuffer === 'function' ? context.isBuffer(buffer) : null });
  }
  function trackModel(label) {
    live();
    if (typeof label !== 'string' || !/^[a-z0-9-]{1,40}$/.test(label)) throw Error('Scalar model label required');
    if (models.some(model => model.label === label)) throw Error('Model label already tracked');
    const found = discover();
    if (!found.complete || found.store !== storeRef.deref() || found.currentModels.length !== 1) throw Error('One current authenticated model required');
    const props = found.currentModels[0].memoizedProps, scene = props.scene;
    if (typeof props.url !== 'string' || !props.url) throw Error('Loaded model URL unavailable');
    if (models.some(model => model.ref.deref() === scene)) throw Error('Model already tracked');
    const model = { label, uuid: scene.uuid, url: props.url, ref: new WeakRef(scene), meshes: [] };
    const nodes = objectTree(scene);
    if (!nodes.complete) throw Error('Model node census incomplete');
    for (const mesh of nodes.rows) {
      if (!mesh.isMesh) continue;
      model.meshes.push({ uuid: mesh.uuid, ref: new WeakRef(mesh) });
      const geometry = mesh.geometry;
      const unseen = geometry && !seen.has(geometry);
      registerResource(geometry, 'geometry', label);
      if (unseen) {
        for (const [key, attribute] of Object.entries(geometry.attributes || {})) trackAttribute(attribute, geometry.uuid, key, label);
        trackAttribute(geometry.index, geometry.uuid, 'index', label);
      }
      for (const material of [mesh.material, mesh.userData?.originalMaterial].flat()) registerResource(material, 'material', label);
    }
    models.push(model);
    return snapshot();
  }
  function objectTree(start) {
    const rows = [], stack = start ? [start] : [], visited = new Set(); let complete = true;
    while (stack.length) {
      const object = stack.pop();
      if (visited.has(object)) { complete = false; continue; }
      if (rows.length >= LIMIT) { complete = false; break; }
      visited.add(object); rows.push(object);
      for (const child of object.children || []) stack.push(child);
    }
    return { rows, complete };
  }
  function cacheState(model) {
    let count = 0; const sceneUuids = [];
    for (const entry of cache) if (entry.keys?.[0] === Loader && entry.keys?.[1] === model.url) {
      count++; const uuid = entry.response?.[0]?.scene?.uuid;
      if (typeof uuid === 'string') sceneUuids.push(uuid);
    }
    return { count, sceneUuids };
  }
  function census() {
    const targets = new Map(), output = { complete: true, current: [], alternate: [], appState: { available: false },
      scope: 'Direct known props/stateNode and hook value/ref/queue/dependencies; no closure or heap reachability proof' };
    const addTarget = (object, row, kind) => { if (object) targets.set(object, { label: row.label, uuid: row.uuid, kind }); };
    for (const model of models) { addTarget(model.ref.deref(), model, 'scene'); for (const mesh of model.meshes) addTarget(mesh.ref.deref(), { ...mesh, label: model.label }, 'mesh'); }
    for (const row of resources) addTarget(row.ref.deref(), row, row.kind);
    const appStates = new Set();
    function check(value, slot, component, group) {
      const target = targets.get(value); if (!target) return;
      if (output[group].length >= 256) { output.complete = false; return; }
      output[group].push({ ...target, component, slot });
    }
    function dependencies(values, slot, component, group) {
      if (!Array.isArray(values)) return;
      if (values.length > 128) output.complete = false;
      for (const value of values.slice(0, 128)) check(value, slot, component, group);
    }
    for (const ref of rootRefs) {
      const root = ref.deref(); if (!root) continue;
      for (const [group, start] of [['current', root.current], ['alternate', root.current?.alternate]]) {
        const tree = walk(start); output.complete = output.complete && tree.complete;
        for (const fiber of tree.rows) {
          const component = name(fiber), props = fiber.memoizedProps;
          for (const key of ['scene', 'object', 'geometry', 'material']) check(props?.[key], 'props.' + key, component, group);
          check(fiber.stateNode, 'stateNode', component, group);
          let hook = fiber.memoizedState, index = 0; const visited = new Set();
          // Hook lists exist on function components; do not inspect root/class state as a hook list.
          if (fiber.tag !== undefined && fiber.tag !== 0 && fiber.tag !== 11 && fiber.tag !== 15) continue;
          while (hook && typeof hook === 'object') {
            if (visited.has(hook) || index++ >= 128) { output.complete = false; break; }
            visited.add(hook); const value = hook.memoizedState;
            check(value, 'hook.value', component, group); check(value?.current, 'hook.ref.current', component, group);
            check(hook.queue?.lastRenderedState, 'hook.queue.lastRenderedState', component, group);
            dependencies(value?.deps, 'hook.effect.dependencies', component, group);
            if (Array.isArray(value)) dependencies(value[1], 'hook.memo.dependencies', component, group);
            if (group === 'current' && component === 'Viewer3D' && value && typeof value === 'object' &&
                Object.hasOwn(value, 'currentJob') && typeof value.setCurrentJob === 'function' &&
                typeof value.meshSelected === 'boolean' && Array.isArray(value.meshHistory)) appStates.add(value);
            hook = hook.next;
          }
        }
      }
    }
    if (appStates.size === 1) {
      const state = appStates.values().next().value, job = state.currentJob;
      output.appState = { available: true, meshSelected: state.meshSelected, historyCount: state.meshHistory.length,
        currentJob: job ? { id: job.id, status: job.status, modelLabels: models.filter(model => model.url === String(state.apiUrl || '') + String(job.outputUrl || '')).map(model => model.label) } : null };
    }
    return output;
  }
  function snapshot() {
    live(); const found = discover(), state = found.store.getState(), gl = glRef.deref(), context = gl?.getContext();
    if (found.store !== storeRef.deref() || state.gl !== gl) throw Error('Authenticated viewer changed');
    const active = objectTree(state.scene), references = census(); references.complete = references.complete && active.complete && found.complete;
    return { time: performance.now(), stopped: false, currentModels: found.currentModels.map(fiber => ({ uuid: fiber.memoizedProps.scene.uuid })),
      renderer: { memory: { ...gl.info.memory }, programs: gl.info.programs?.length, render: { ...gl.info.render } },
      models: models.map(model => {
        const object = model.ref.deref(); let parent = object, attached = false, steps = 0;
        while (parent && steps++ < LIMIT) { if (parent === state.scene) { attached = true; break; } parent = parent.parent; }
        if (steps >= LIMIT) references.complete = false;
        const objects = new Set([object, ...model.meshes.map(mesh => mesh.ref.deref())].filter(Boolean));
        const ownedResources = resources.filter(resource => resource.labels.includes(model.label)).map(resource => resource.ref.deref()).filter(Boolean);
        const activeSceneMatches = active.rows.filter(node => objects.has(node) || ownedResources.includes(node.geometry) ||
          [node.material].flat().some(material => material && ownedResources.includes(material))).length;
        return { label: model.label, uuid: model.uuid, alive: !!object, attachedToMainScene: attached, activeSceneMatches, cache: cacheState(model) };
      }),
      resources: resources.map(row => { const object = row.ref.deref(); return { label: row.label, labels: [...row.labels], kind: row.kind, uuid: row.uuid,
        alive: !!object, disposals: row.disposals, lastDisposeAt: row.lastDisposeAt,
        bvhPresent: row.kind === 'geometry' && object ? !!object.boundsTree : null, loadedPropertyPresent: row.loadedPropertyPresent,
        propertyPresent: row.kind === 'material' && object && typeof gl.properties?.has === 'function' ? gl.properties.has(object) : null }; }),
      attributes: attributeRecords.map(row => { const attribute = row.ref.deref(), buffer = row.bufferRef?.deref(); return {
        label: row.label, geometryUuid: row.geometryUuid, name: row.name, attributeAlive: !!attribute, bufferAlive: !!buffer,
        loadedRecordPresent: row.loadedRecordPresent, loadedBufferValid: row.loadedBufferValid,
        recordPresent: attribute ? !!attributes.get(attribute) : null,
        bufferValid: buffer && typeof context?.isBuffer === 'function' ? context.isBuffer(buffer) : null }; }), census: references };
  }
  function stop() {
    if (stopped) return receipt;
    let listenersRemoved = 0;
    for (const row of resources) { const object = row.ref.deref(); if (object) { object.removeEventListener('dispose', row.listener); listenersRemoved++; } }
    receipt = { stopped: true, listenersRemoved, rootHandlesReleased: rootRefs.length, modelRecordsReleased: models.length,
      resourceRecordsReleased: resources.length, attributeRecordsReleased: attributeRecords.length, cacheReleased: true, attributesReleased: true };
    stopped = true; rootRefs = []; models = []; resources = []; attributeRecords = []; seen = null;
    cache = null; Loader = null; attributes = null; storeRef = null; glRef = null;
    return receipt;
  }
  return { trackModel, snapshot, stop };
})
