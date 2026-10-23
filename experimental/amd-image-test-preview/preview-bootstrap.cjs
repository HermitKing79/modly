// Generated configuration is immutable; this template never selects a user profile.
'use strict';
const { app, ipcMain, session } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const options = __MODLY_PREVIEW_OPTIONS__;
const identity = value => process.platform === 'win32' ? path.resolve(value).toLowerCase() : path.resolve(value);
const within = (root, value) => {
  const relative = path.relative(root, value);
  return relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative);
};
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function start() {
  const profile = path.resolve(options.profile);
  const nonce = process.env.MODLY_AMD_PREVIEW_NONCE || '';
  const file = process.env.MODLY_AMD_PREVIEW_ATTACH_FILE || '';
  const relative = path.relative(profile, path.resolve(file)).split(path.sep);
  if (!/^[a-f0-9]{32}$/.test(nonce) || !path.isAbsolute(file) || relative.length !== 3 ||
      relative[0] !== 'launches' || !/^[a-zA-Z0-9-]+$/.test(relative[1]) || relative[2] !== 'attached.json' ||
      identity(fs.realpathSync(profile)) !== identity(profile)) {
    throw Error('Use Launch AMD Image Test.cmd: a private launcher attachment is required');
  }
  const ensureDirectory = directory => {
    let ancestor = directory;
    while (!fs.existsSync(ancestor)) ancestor = path.dirname(ancestor);
    if (!within(profile, fs.realpathSync(ancestor))) throw Error('Private path ancestor escaped profile');
    fs.mkdirSync(directory, { recursive: true });
    if (!within(profile, fs.realpathSync(directory))) throw Error('Private directory escaped profile');
    return directory;
  };
  const directories = { appData: 'app-data', userData: '', sessionData: 'session', temp: 'temp',
    logs: 'logs', crashDumps: 'crash-dumps', home: 'home', desktop: 'home/Desktop', documents: 'documents', downloads: 'downloads' };
  for (const [key, relative] of Object.entries(directories)) {
    const directory = path.join(profile, relative);
    ensureDirectory(directory);
    app.setPath(key, directory);
  }
  for (const key of ['PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'VIRTUAL_ENV', 'CONDA_PREFIX',
    'PIP_USER', 'PIP_TARGET', 'PIP_PREFIX', 'NODE_OPTIONS', 'ELECTRON_RENDERER_URL', 'ELECTRON_RUN_AS_NODE',
    'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN', 'CHROME_LOG_FILE', 'ELECTRON_LOG_FILE']) delete process.env[key];
  for (const [key, relative] of Object.entries({ TEMP: 'temp', TMP: 'temp', HOME: 'home', USERPROFILE: 'home',
    APPDATA: 'app-data', LOCALAPPDATA: 'local-app-data', XDG_CACHE_HOME: 'cache', XDG_CONFIG_HOME: 'config',
    HF_HOME: 'cache/huggingface', HUGGINGFACE_HUB_CACHE: 'cache/huggingface/hub', TRANSFORMERS_CACHE: 'cache/transformers',
    TORCH_HOME: 'cache/torch', PIP_CACHE_DIR: 'cache/pip', NUMBA_CACHE_DIR: 'cache/numba',
    MPLCONFIGDIR: 'cache/matplotlib', TRITON_CACHE_DIR: 'cache/triton' })) {
    const directory = path.join(profile, relative);
    ensureDirectory(directory);
    process.env[key] = directory;
  }
  Object.assign(process.env, { MODLY_AMD_IMAGE_TEST_PYTHON: options.cpuPython,
    PYTHONDONTWRITEBYTECODE: '1', PYTHONNOUSERSITE: '1', HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1',
    CUDA_VISIBLE_DEVICES: '', HIP_VISIBLE_DEVICES: '', ROCR_VISIBLE_DEVICES: '', SELECTED_MODEL_ID: '' });
  for (const relative of ['models','workspace','workflows','extensions','dependencies','agent','builtin-extensions',
    'cache/chromium','cache/media']) ensureDirectory(path.join(profile, relative));
  process.env.CHROME_LOG_FILE = path.join(profile, 'logs/chromium.log');
  process.env.ELECTRON_LOG_FILE = path.join(profile, 'logs/electron.log');
  app.setName('Noncanonical AMD Image Test Preview');
  app.disableHardwareAcceleration();
  for (const flag of ['disable-gpu', 'disable-gpu-compositing', 'disable-background-networking',
    'disable-component-update', 'disable-domain-reliability']) app.commandLine.appendSwitch(flag);
  app.commandLine.appendSwitch('disk-cache-dir', path.join(profile, 'cache/chromium'));
  app.commandLine.appendSwitch('media-cache-dir', path.join(profile, 'cache/media'));
  app.commandLine.appendSwitch('log-file', path.join(profile, 'logs/chromium.log'));
  app.commandLine.appendSwitch('lang', 'en-US');

  // Storage and GPU flags above run synchronously, before the first possible await.
  const deadline = Date.now() + 10000;
  while (true) {
    if (fs.existsSync(file)) {
      if (!within(profile, fs.realpathSync(file))) throw Error('Launcher attachment escaped the private profile');
      const record = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (record.nonce !== nonce || record.attached !== true) throw Error('Launcher attachment identity mismatch');
      break;
    }
    if (Date.now() >= deadline) throw Error('Launcher attachment timed out; no preview main process was started');
    await delay(50);
  }

  const handles = new Set(['setup:check', 'python:start', 'python:status', 'app:info', 'settings:get',
    'fs:selectImage', 'fs:readFileBase64', 'fs:readScreenshotDataUrl', 'fs:listDir', 'fs:listFiles',
    'window:isMaximized', 'system:memory', 'log:getPath', 'log:readAll', 'log:listSessions',
    'workflows:list', 'extensions:list', 'workspace:listCollections', 'workspace:listJobs',
    'workspace:library:list', 'workspace:library:read']);
  const events = new Set(['window:minimize', 'window:maximize', 'window:close', 'log:error']);
  const handle = ipcMain.handle.bind(ipcMain);
  const on = ipcMain.on.bind(ipcMain);
  ipcMain.handle = (channel, listener) => handle(channel, channel === 'extensions:list' ? async () => [] : handles.has(channel) ? listener : async () => {
    throw Error('Preview blocks IPC: ' + channel);
  });
  ipcMain.on = (channel, listener) => on(channel, events.has(channel) ? listener : () => {});

  await app.whenReady();
  session.defaultSession.webRequest.onBeforeRequest({ urls: ['http://*/*', 'https://*/*', 'ws://*/*', 'wss://*/*'] },
    (details, callback) => {
      let allowed = false;
      try {
        const url = new URL(details.url);
        allowed = url.origin === 'http://127.0.0.1:8765' && !url.username && !url.password && !url.search &&
          ((details.method === 'GET' && url.pathname === '/health') ||
           (details.method === 'POST' && url.pathname === '/amd/image-test'));
      } catch {}
      callback({ cancel: !allowed });
    });
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.on('will-download', event => event.preventDefault());
  app.on('web-contents-created', (_event, contents) => {
    contents.on('will-navigate', event => event.preventDefault());
    contents.on('will-attach-webview', event => event.preventDefault());
    contents.setWindowOpenHandler(() => ({ action: 'deny' }));
  });
  require('./index.js');
}

start().catch(error => {
  try { process.stderr.write('[AMD Image Test Preview] ' + String(error.message || error) + '\n'); } catch {}
  app.exit(1);
});
