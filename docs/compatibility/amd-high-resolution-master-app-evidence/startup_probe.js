// Function expression for Debugger.evaluateOnCallFrame at the paused installed main.
// Root invokes: `(${source})(${JSON.stringify(CONFIG)})`; never run standalone.
(function isolateInstalledModly(CONFIG) {
  'use strict';
  const fs = require('fs'), path = require('path'), { app, session } = require('electron');
  const need = (ok, message) => { if (!ok) throw new Error(message); };
  need(CONFIG && Object.keys(CONFIG).sort().join(',') === 'mainFile,privateRoot,profileDir,sessionDir', 'Exact isolation configuration');
  const normal = p => path.resolve(p).toLowerCase();
  need(typeof __filename === 'string' && normal(__filename) === normal(CONFIG.mainFile), 'Exact paused installed main module');
  need(!app.isReady() && !globalThis.__modlyAcceptance, 'Pause before app readiness; no repeated isolation');
  function plainDirectory(raw) {
    need(typeof raw === 'string' && path.isAbsolute(raw), 'Absolute precreated isolation directory');
    const resolved = path.resolve(raw); let cursor = resolved;
    for (;;) {
      const info = fs.lstatSync(cursor);
      need(info.isDirectory() && !info.isSymbolicLink(), 'Plain directory and ancestors');
      const parent = path.dirname(cursor); if (parent === cursor) break; cursor = parent;
    }
    need(normal(fs.realpathSync(resolved)) === normal(resolved), 'No redirected isolation directory');
    return resolved;
  }
  const root = plainDirectory(CONFIG.privateRoot);
  const profile = plainDirectory(CONFIG.profileDir), storage = plainDirectory(CONFIG.sessionDir);
  const downloads = plainDirectory(path.join(root, 'exports'));
  for (const directory of [profile, storage, downloads]) {
    const rel = path.relative(root, directory);
    need(rel && rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel), 'Isolation inside private root');
  }
  need(new Set([profile, storage, downloads].map(normal)).size === 3, 'Distinct private directories');
  app.setPath('userData', profile); app.setPath('sessionData', storage); app.setPath('downloads', downloads);
  process.env.PYTHONDONTWRITEBYTECODE = '1';
  const state = { started: new Date().toISOString(), calls: [], errors: [], downloads: [], windows: [], overflow: 0 };
  const add = (list, row) => { if (list.length < 256) list.push(row); else state.overflow++; };
  const updater = require('electron-updater').autoUpdater, denied = {};
  for (const name of ['checkForUpdates', 'downloadUpdate', 'quitAndInstall', 'install']) {
    need(typeof updater[name] === 'function', 'Installed updater method: ' + name);
    denied[name] = function () {
      add(state.calls, { method: name, time: Date.now() });
      if (name === 'checkForUpdates') return Promise.resolve(null);
      if (name === 'downloadUpdate') return Promise.reject(new Error('Diagnostic update download denied'));
      throw new Error('Diagnostic update installation denied');
    };
    Object.defineProperty(updater, name, { value: denied[name], writable: false, configurable: false });
  }
  Object.defineProperty(updater, 'autoInstallOnAppQuit', { enumerable: true, configurable: false,
    get: () => false, set: value => add(state.calls, { method: 'autoInstallOnAppQuit:set', value, time: Date.now() }) });
  app.on('browser-window-created', (_event, window) => {
    const wc = window.webContents, row = { id: window.id, webContentsId: wc.id }; add(state.windows, row);
    wc.on('console-message', (_event, detail, message, line, source) => {
      const d = detail && typeof detail === 'object' ? detail : { level: detail, message, lineNumber: line, sourceId: source };
      if (d.level === 'error' || d.level === 'warning' || (typeof d.level === 'number' && d.level >= 2))
        add(state.errors, { type: 'console', window: row.id, level: d.level, message: String(d.message).slice(0, 4096), line: d.lineNumber, source: d.sourceId });
    });
    wc.on('did-fail-load', (_e, code, description, url, mainFrame) => add(state.errors, { type: 'load', code, description, url, mainFrame }));
    wc.on('render-process-gone', (_e, detail) => add(state.errors, { type: 'renderer-gone', detail }));
    wc.on('preload-error', (_e, script, error) => add(state.errors, { type: 'preload', script, message: String(error) }));
    window.on('unresponsive', () => add(state.errors, { type: 'unresponsive', window: row.id }));
  });
  app.whenReady().then(() => session.defaultSession.on('will-download', (_event, item) => {
    const row = { url: item.getURL(), filename: item.getFilename(), totalBytes: item.getTotalBytes(), started: Date.now(), state: 'started' };
    add(state.downloads, row);
    item.on('updated', (_e, status) => { row.state = status; row.receivedBytes = item.getReceivedBytes(); });
    item.once('done', (_e, status) => { row.state = status; row.receivedBytes = item.getReceivedBytes(); row.savePath = item.getSavePath(); row.ended = Date.now(); });
  })).catch(error => add(state.errors, { type: 'ready-handler', message: String(error) }));
  const snapshot = () => JSON.parse(JSON.stringify({ ...state, appReady: app.isReady(),
    userData: app.getPath('userData'), sessionData: app.getPath('sessionData'), downloadsPath: app.getPath('downloads'),
    updaterMethodsIntact: Object.keys(denied).every(name => updater[name] === denied[name]), autoInstallOnAppQuit: updater.autoInstallOnAppQuit,
    mainMemory: process.memoryUsage(), appMetrics: app.isReady() ? app.getAppMetrics() : [] }));
  Object.defineProperty(globalThis, '__modlyAcceptance', { value: Object.freeze({ snapshot }), configurable: false, writable: false });
  need(normal(app.getPath('userData')) === normal(profile) && normal(app.getPath('sessionData')) === normal(storage)
    && normal(app.getPath('downloads')) === normal(downloads), 'Effective private Electron paths');
  return snapshot();
})
