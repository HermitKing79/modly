(function () {
  'use strict';
  if (globalThis.__modlyObserver !== undefined) throw new Error('observer already installed');
  const state = globalThis.__modlyObserver = {
    installed: false, restored: false, ready: false, calls: 0, errors: []
  };
  let prototype, original;
  try {
    prototype = require('electron').BrowserWindow.prototype;
    original = prototype.loadFile;
    if (typeof original !== 'function') throw new Error('BrowserWindow.loadFile unavailable');
  } catch (error) {
    state.errors.push(String(error.message || error));
    return state;
  }
  const source = `(() => {
    if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__ !== undefined || window.__modlyRoots !== undefined)
      throw new Error('modly observer collision');
    const roots = new Set(); let next = 0;
    window.__modlyRoots = roots;
    window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
      supportsFiber: true, isDisabled: false, inject() { return ++next; }, checkDCE() {},
      onCommitFiberRoot(id, root) {
        if (root && root.current && root.current.child) roots.add(root); else roots.delete(root);
      },
      onCommitFiberUnmount(id, fiber) { if (fiber && fiber.tag === 3) roots.delete(fiber.stateNode); },
      onPostCommitFiberRoot() {}
    };
  })();`;
  async function observedLoadFile(...args) {
    state.calls++;
    let debuggerApi, attached = false;
    try {
      if (state.calls !== 1 || prototype.loadFile !== observedLoadFile)
        throw new Error('loadFile observer ownership changed');
      prototype.loadFile = original;
      state.restored = true;
      debuggerApi = this.webContents.debugger;
      if (debuggerApi.isAttached()) throw new Error('preexisting renderer debugger');
      debuggerApi.attach('1.3'); attached = true;
      await debuggerApi.sendCommand('Page.enable');
      const registration = await debuggerApi.sendCommand('Page.addScriptToEvaluateOnNewDocument', { source });
      if (!registration || typeof registration.identifier !== 'string' || !registration.identifier)
        throw new Error('new-document observer registration unavailable');
      state.ready = true; // Registration only; caller must separately observe actual committed roots.
      return await original.apply(this, args);
    } catch (error) {
      state.ready = false;
      state.errors.push(String(error.message || error));
      if (prototype.loadFile === observedLoadFile) {
        prototype.loadFile = original; state.restored = true;
      }
      if (attached) {
        try { debuggerApi.detach(); } catch (cleanup) { state.errors.push(String(cleanup.message || cleanup)); }
      }
      throw error;
    }
  }
  prototype.loadFile = observedLoadFile;
  state.installed = true;
  return state;
})
