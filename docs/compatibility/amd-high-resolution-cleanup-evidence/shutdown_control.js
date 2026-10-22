(function (CONFIG) {
  'use strict';
  const { app, BrowserWindow } = require('electron'), fs = require('fs'), path = require('path');
  if (globalThis.__modlyCleanupShutdown || app.isReady()) throw Error('One shutdown control before readiness');
  if (!(CONFIG.startedEpoch > 0 && CONFIG.shutdownEpoch > CONFIG.startedEpoch &&
      CONFIG.shutdownEpoch - CONFIG.startedEpoch <= 221)) throw Error('Shutdown reserve at220seconds');
  let requested = false, restored = false, timer;
  const save = (name, value) => fs.writeFileSync(path.join(CONFIG.privateRoot, name), JSON.stringify(value, null, 2)+'\n', { flag: 'wx' });
  const stamp = () => ({ at: new Date().toISOString(), elapsedSeconds: Date.now()/1000-CONFIG.startedEpoch });
  const restore = async () => {
    if (restored) return { alreadyRestored: true };
    const windows = BrowserWindow.getAllWindows();
    if (windows.length !== 1) throw Error('Exact owned window for restoration');
    const wc = windows[0].webContents, observer = globalThis.__modlyObserver;
    const renderer = await wc.executeJavaScript(`(() => {
      const owned = window.__REACT_DEVTOOLS_GLOBAL_HOOK__;
      if (owned && owned.__modlyCleanupOwned !== true) throw Error('Foreign React observer');
      const stopped = window.__modlyCleanup ? window.__modlyCleanup.stop() : { notInstalled: true };
      if (owned) {
        owned.onCommitFiberRoot = () => {}; owned.onCommitFiberUnmount = () => {};
        owned.onPostCommitFiberRoot = () => {};
      }
      window.__modlyRoots?.clear();
      delete window.__modlyCleanup; delete window.__modlyRoots;
      delete window.__REACT_DEVTOOLS_GLOBAL_HOOK__; delete window.__modlyCleanupStore;
      return { stopped, globalsRemoved: !('__modlyCleanup' in window) && !('__modlyRoots' in window) &&
        !('__REACT_DEVTOOLS_GLOBAL_HOOK__' in window) && !('__modlyCleanupStore' in window) };
    })()`, false);
    if (!observer?.restored || observer.webContentsId !== wc.id || !observer.registrationIdentifier || !observer.debuggerAttached || !wc.debugger.isAttached())
      throw Error('Exact owned renderer debugger/registration');
    await wc.debugger.sendCommand('Page.removeScriptToEvaluateOnNewDocument', { identifier: observer.registrationIdentifier });
    await wc.debugger.sendCommand('Runtime.releaseObjectGroup', { objectGroup: 'modly-cleanup-qualification' });
    wc.debugger.detach(); observer.debuggerAttached = false;
    restored = renderer.globalsRemoved && !wc.debugger.isAttached();
    const receipt = { ...stamp(), restored, renderer, registrationRemoved: true, rendererDebuggerDetached: !wc.debugger.isAttached(),
      loadFileWrapperRestored: observer.restored };
    save('observer-restoration.json', receipt);
    return receipt;
  };
  const finish = async reason => {
    if (requested) return { requested: true, restored };
    requested = true; clearTimeout(timer);
    save('shutdown-request.json', { ...stamp(), reason, mechanism: 'Electron app.quit', supervisorForce: false });
    let restoration;
    try {
      restoration = await Promise.race([restore(), new Promise((_, no) => setTimeout(() => no(Error('1second restoration bound')), 1000))]);
    } catch (error) {
      restoration = { restored: false, error: String(error) };
      save('observer-restoration-failure.json', { ...stamp(), ...restoration });
    }
    // Allow the inspector to return the scalar receipt before normal app quit.
    setTimeout(() => app.quit(), 50);
    return { requested: true, restoration };
  };
  app.once('before-quit', () => save('app-before-quit.json', { ...stamp(), requested, restored }));
  app.once('will-quit', () => save('app-will-quit.json', { ...stamp(), requested, restored }));
  timer = setTimeout(() => { void finish('scheduled-220second-reserve'); }, Math.max(0, CONFIG.shutdownEpoch*1000-Date.now()));
  globalThis.__modlyCleanupShutdown = Object.freeze({ finish, restore,
    snapshot: () => ({ requested, restored, shutdownEpoch: CONFIG.shutdownEpoch, ...stamp() }) });
  return globalThis.__modlyCleanupShutdown.snapshot();
})
