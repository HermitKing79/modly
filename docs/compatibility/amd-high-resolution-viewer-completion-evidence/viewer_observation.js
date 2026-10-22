(function (roots) {
  'use strict';
  // Passive, temporary renderer observation. No imports, scene writes, cache calls,
  // input synthesis, asset copies, forced GC, or changes to controls.
  function walk(start) {
    const out = [], stack = start ? [start] : [];
    while (stack.length) {
      const f = stack.pop(); out.push(f);
      if (f.sibling) stack.push(f.sibling);
      if (f.child) stack.push(f.child);
    }
    return out;
  }
  function name(f) { return f.type && (f.type.name || f.type.displayName); }
  function discover() {
    const matches = [];
    for (const root of roots || []) {
      const fibers = walk(root.current);
      const captures = fibers.filter(f => name(f) === 'CanvasCapture');
      for (const capture of captures) {
        const canvas = capture.memoizedProps?.domRef?.current;
        if (!canvas || canvas.tagName !== 'CANVAS') continue;
        for (const provider of fibers) {
          const store = provider.memoizedProps?.store;
          if (typeof store?.getState !== 'function') continue;
          const s = store.getState(), c = s.controls;
          if (s.gl?.domElement !== canvas || !s.scene || !s.camera || !c ||
              c.object !== s.camera || typeof c.addEventListener !== 'function') continue;
          const input = s.events?.connected || canvas;
          if (c.domElement !== input || !(input === canvas || input.contains?.(canvas))) continue;
          const defaults = fibers.filter(f => f.memoizedProps?.makeDefault === true &&
            walk(f.child).some(n => n.memoizedProps?.object === c && n.stateNode === c));
          if (!defaults.length) continue;
          const models = fibers.filter(f => name(f) === 'SceneMeshModel');
          if (models.some(f => f.memoizedProps?.scene?.__r3f?.root !== store)) continue;
          if (!matches.some(m => m.store === store)) matches.push({ store, canvas, controls: c, input,
            models: models.map(f => f.memoizedProps.scene) });
        }
      }
    }
    if (matches.length !== 1) throw Error('Expected one authenticated main viewer; got ' + matches.length);
    return matches[0];
  }
  const found = discover(), storeRef = new WeakRef(found.store), gl = found.store.getState().gl;
  const mainSceneRef = new WeakRef(found.store.getState().scene);
  const cameraRef = new WeakRef(found.store.getState().camera);
  const controlsRef = new WeakRef(found.controls), inputRef = new WeakRef(found.input);
  // Do not retain found.models (or a React fiber, root state, or old scene) in API closures.
  found.models = null; found.store = null; found.controls = null; found.input = null; found.canvas = null;
  const resources = [], models = [], seen = new WeakMap();
  let phase = null, stopped = false, lastRender = null, renders = 0, sequence = 0;
  let frameTick = 0, lastFrame = null, lastRenderedTick = null, lastRenderedFrame = null;
  const now = () => performance.now();
  // Priority zero does not take over rendering. OrbitControls runs at -1;
  // EffectComposer renders later. Number actual R3F ticks, never browser RAFs.
  function onFrame() {
    frameTick++;
    if (!phase) return;
    const t = now();
    if (lastFrame !== null && phase.frameIntervals.length < 4096) phase.frameIntervals.push(t-lastFrame);
    lastFrame = t; phase.frameTicks++;
  }
  const unsubscribeFrame = storeRef.deref().getState().internal.subscribe({ current: onFrame }, 0, storeRef.deref());
  function pose() {
    const c = controlsRef.deref(), a = cameraRef.deref();
    return a && c ? [a.position.x, a.position.y, a.position.z,
      a.quaternion.x, a.quaternion.y, a.quaternion.z, a.quaternion.w,
      c.target.x, c.target.y, c.target.z, a.zoom] : null;
  }
  function differs(a, b) { return a && b && a.some((v, i) => Math.abs(v - b[i]) > 1e-10); }
  function onInput(e) {
    const input = inputRef.deref();
    if (!phase || !e.isTrusted || !input || !e.composedPath().includes(input)) return;
    if (e.type !== 'wheel' && !(e.type === 'pointermove' && e.buttons)) return;
    if (phase.inputs.length >= 1024) { phase.truncated = true; return; }
    phase.inputs.push({ id: ++sequence, type: e.type, at: now(), before: pose(),
      button: e.button, buttons: e.buttons, ctrl: e.ctrlKey, shift: e.shiftKey,
      meta: e.metaKey, deltaY: e.type === 'wheel' ? e.deltaY : null,
      changeAt: null, renderAt: null, renderReturnAt: null, renderFrameTick: null });
  }
  function onChange() {
    if (!phase || !phase.inputs.length) return;
    const p = phase.inputs[phase.inputs.length - 1];
    if (p.changeAt === null && differs(p.before, pose())) p.changeAt = now();
  }
  const original = gl.render;
  function observedRender(...args) {
    const main = args[0] === mainSceneRef.deref() && args[1] === cameraRef.deref();
    const start = main ? now() : null;
    const result = original.apply(this, args);
    if (main) {
      const end = now(); renders++;
      if (phase) {
        if (lastRender !== null && phase.mainRenderIntervals.length < 4096) phase.mainRenderIntervals.push(start - lastRender);
        if (lastRenderedTick !== frameTick) {
          if (lastRenderedFrame !== null && phase.renderedFrameIntervals.length < 4096)
            phase.renderedFrameIntervals.push(start-lastRenderedFrame);
          lastRenderedFrame = start; lastRenderedTick = frameTick; phase.renderedFrameTicks++;
        }
        for (const p of phase.inputs) if (p.changeAt !== null && p.renderAt === null && start >= p.changeAt) {
          p.renderAt = start; p.renderReturnAt = end; p.renderFrameTick = frameTick;
        }
        phase.mainRenderCalls++;
      }
      lastRender = start;
    }
    return result;
  }
  gl.render = observedRender;
  window.addEventListener('pointermove', onInput, true);
  window.addEventListener('wheel', onInput, { capture: true, passive: true });
  controlsRef.deref().addEventListener('change', onChange);
  function register(object, kind, label) {
    if (!object || seen.has(object)) return;
    const r = { label, kind, uuid: object.uuid, ref: new WeakRef(object), disposals: 0, lastDisposeAt: null };
    const listener = () => { r.disposals++; r.lastDisposeAt = now(); };
    r.listener = listener; seen.set(object, r); resources.push(r);
    object.addEventListener('dispose', listener);
  }
  function trackModel(label) {
    const d = discover();
    if (d.store !== storeRef.deref() || d.models.length !== 1) throw Error('One current main model required');
    const model = d.models[0];
    if (models.some(m => m.ref.deref() === model)) throw Error('Model already tracked');
    models.push({ label, uuid: model.uuid, ref: new WeakRef(model) });
    model.traverse(child => {
      if (!child.isMesh) return;
      register(child.geometry, 'geometry', label);
      for (const material of [child.material, child.userData?.originalMaterial].flat()) {
        if (!material) continue;
        register(material, 'material', label);
        for (const value of Object.values(material)) if (value?.isTexture) register(value, 'texture', label);
      }
    });
    return snapshot();
  }
  function multiply(m, p) {
    return [m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12]*p[3],
      m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13]*p[3],
      m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14]*p[3],
      m[3]*p[0]+m[7]*p[1]+m[11]*p[2]+m[15]*p[3]];
  }
  function corners(min, max) {
    return Array.from({length: 8}, (_, i) => [i&1 ? max[0] : min[0], i&2 ? max[1] : min[1], i&4 ? max[2] : min[2], 1]);
  }
  function framing(model, camera) {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    let count = 0, unavailable = false;
    model.traverse(child => {
      if (!child.isMesh) return;
      const box = child.geometry?.boundingBox, m = child.matrixWorld?.elements;
      if (!box || !m) { unavailable = true; return; }
      for (const p of corners([box.min.x,box.min.y,box.min.z], [box.max.x,box.max.y,box.max.z])) {
        const w = multiply(m,p); count++;
        for(let i=0;i<3;i++){min[i]=Math.min(min[i],w[i]/w[3]);max[i]=Math.max(max[i],w[i]/w[3]);}
      }
    });
    if (unavailable || !count) return { available: false };
    const ndc = corners(min,max).map(p => {
      const q = multiply(camera.projectionMatrix.elements, multiply(camera.matrixWorldInverse.elements,p));
      return [q[0]/q[3],q[1]/q[3],q[2]/q[3],q[3]];
    });
    return { available: true, definition: 'eight corners of conservative world axis-aligned model bounds; no matrix updates or bounding-box computation',
      boundsWorld: { min, max }, cornersNdc: ndc,
      allCornersInside: ndc.every(p => p[3]>0 && p.slice(0,3).every(v => v>=-1 && v<=1)),
      viewportHeightFraction: (Math.max(...ndc.map(p=>p[1]))-Math.min(...ndc.map(p=>p[1])))/2 };
  }
  function snapshot() {
    const d = discover(), s = d.store.getState();
    const control = d.controls;
    return { time: now(), stopped, currentModels: d.models.map(m=>({uuid:m.uuid,
      transform: { position: [m.position.x,m.position.y,m.position.z],
        quaternion: [m.quaternion.x,m.quaternion.y,m.quaternion.z,m.quaternion.w],
        scale: [m.scale.x,m.scale.y,m.scale.z] }, framing:framing(m,s.camera)})),
      pose: pose(), mainRenderCalls: renders, frameTick, cameraType: s.camera.type,
      controls: { enabled: control.enabled, enablePan: control.enablePan, enableZoom: control.enableZoom,
        enableRotate: control.enableRotate, autoRotate: control.autoRotate, mouseButtons: {...control.mouseButtons},
        keyboardListenerConnected: !!control._domElementKeyEvents },
      renderer: { memory: {...s.gl.info.memory}, programs: s.gl.info.programs?.length,
        render: {...s.gl.info.render} },
      models: models.map(m=>{ const o=m.ref.deref();return {label:m.label,uuid:m.uuid,alive:!!o,
        attachedToMainScene:!!o && (function(){let p=o;while(p){if(p===s.scene)return true;p=p.parent;}return false;})()}; }),
      resources: resources.map(r=>{ const o=r.ref.deref();return {label:r.label,kind:r.kind,uuid:r.uuid,alive:!!o,
        disposals:r.disposals,lastDisposeAt:r.lastDisposeAt,bvhPresent:r.kind==='geometry'&&o?!!o.boundsTree:null}; }) };
  }
  return {
    snapshot, trackModel,
    beginWindow(label) {
      if (phase) throw Error('Observation window already active');
      lastRender = null; lastFrame = null; lastRenderedTick = null; lastRenderedFrame = null;
      phase = { label, begin: now(), initialPose: pose(), inputs: [],
        frameIntervals: [], mainRenderIntervals: [], renderedFrameIntervals: [],
        frameTicks: 0, renderedFrameTicks: 0, mainRenderCalls: 0, truncated: false };
      return snapshot();
    },
    endWindow() {
      if (!phase) throw Error('No observation window');
      const p = phase; phase = null;
      return { ...p, end: now(), finalPose: pose(), snapshot: snapshot(),
        measurement: 'frameIntervals: actual R3F priority-zero tick cadence; renderedFrameIntervals: first successful main render per R3F tick; mainRenderIntervals: all successful main render calls. Trusted event receipt to controls pose change and subsequent main scene/camera renderer.render entry/return is CPU observation, not OS latency or GPU/display presentation' };
    },
    stop() {
      if (stopped) return;
      stopped = true; phase = null;
      unsubscribeFrame();
      if (gl.render === observedRender) gl.render = original;
      window.removeEventListener('pointermove', onInput, true);
      window.removeEventListener('wheel', onInput, true);
      controlsRef.deref()?.removeEventListener('change', onChange);
      for (const r of resources) r.ref.deref()?.removeEventListener('dispose', r.listener);
    }
  };
})
