// Developer-only builder: existing local runtimes, exclusive fresh destination, no app launch.
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const BASE = '63332ca5d415de8eaaff15fdaf193bc7d9935a72'
const SCHEMA = 'modly.amd.image-test-preview.v1'
const API_PYTHON = 'C:/Users/wattz/Documents/Modly/dependencies/venv/Scripts/python.exe'
const CPU_PYTHON = 'D:/AI/Trellis2AMD/venv/Scripts/python.exe'
const QUALIFIED_API_SHA = '5b9eb768cd084d4b7959bc83b1a5111a10f8aaa18e5f4f7ef4187baf85dcf2c3'
const QUALIFIED_CPU_SHA = '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'
const KEYS = ['repo', 'seedRuntime', 'destination', 'apiPython', 'cpuPython']
const sha = value => crypto.createHash('sha256').update(value).digest('hex')
const fileSha = filename => sha(fs.readFileSync(filename))
const put = (filename, value) => {
  fs.mkdirSync(path.dirname(filename), { recursive: true })
  fs.writeFileSync(filename, value)
}
const contains = (root, filename) => {
  const relative = path.relative(root, filename)
  return relative === '' || (relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative))
}

export function createPlan(options) {
  if (Object.keys(options).some(key => !KEYS.includes(key)) || KEYS.some(key => typeof options[key] !== 'string' || !path.isAbsolute(options[key]))) {
    throw Error('Exactly repo, seedRuntime, destination, apiPython and cpuPython absolute paths are required')
  }
  const plan = Object.fromEntries(KEYS.map(key => [key, path.resolve(options[key])]))
  for (const other of [plan.repo, plan.seedRuntime]) {
    if (contains(other, plan.destination) || contains(plan.destination, other)) throw Error('Preview must be separate from source and seed')
  }
  if (fs.existsSync(plan.destination)) throw Error('Fresh preview destination already exists; preserve it and select a new destination')
  return { ...plan, profile: path.join(plan.destination, 'profile'), apiDir: path.join(plan.destination, 'api') }
}

function replaceFunction(source, signature, body) {
  const at = source.indexOf(signature)
  if (at < 0 || source.indexOf(signature, at + signature.length) >= 0) throw Error('Guard signature mismatch: ' + signature)
  const begin = source.indexOf('{', at + signature.length)
  let depth = 1, end = begin + 1
  for (; end < source.length && depth; end++) {
    if (source[end] === '{') depth++
    if (source[end] === '}') depth--
  }
  if (depth) throw Error('Guard body mismatch: ' + signature)
  return source.slice(0, begin + 1) + '\n' + body + '\n}' + source.slice(end)
}

export function guardedSources(options) {
  const profile = options.profile ?? path.join(options.destination, 'profile')
  const apiDir = path.join(options.destination, 'api')
  const literal = value => JSON.stringify(value.replaceAll('\\', '/'))
  const result = new Map()
  result.set('updater', 'export const updatesSupported = false\nexport function initAutoUpdater() {}\n')
  result.set('builtin-sync', `import {app} from 'electron'\nimport {join} from 'path'\nexport function getBuiltinExtensionsDir(){return join(app.getPath('userData'),'builtin-extensions')}\nexport function syncBuiltinExtensions(){}\n`)
  result.set('settings-store', `import {join} from 'path'\nimport {mkdirSync} from 'fs'\nconst profile=${literal(profile)}\nconst settings=Object.freeze(Object.fromEntries(['models','workspace','workflows','extensions','dependencies','agent'].map(name=>[name+'Dir',join(profile,name)])))\nexport function getSettings(_userData:string):any{return {...settings}}\nexport function setSettings():never{throw Error('Preview settings writes are disabled')}\nexport function ensureAgentDir(_userData:string):string{const directory=join(profile,'agent');mkdirSync(directory,{recursive:true});return directory}\n`)
  result.set('hf-token', `export function getHfToken(){return ''}\nexport function initHfToken(){return ''}\nexport function setHfToken():never{throw Error('Preview token writes are disabled')}\n`)
  result.set('python-setup', `const executable=${literal(options.apiPython)}\nconst forbidden=():never=>{throw Error('Preview setup/install is disabled')}\nexport function cleanPythonEnv(){const env={...process.env};for(const key of ['PYTHONHOME','PYTHONPATH','PYTHONSTARTUP','PYTHONUSERBASE','PIP_USER','PIP_TARGET','PIP_PREFIX','VIRTUAL_ENV','CONDA_PREFIX'])delete env[key];return env}\nexport function checkSetupNeeded(){return false}\nexport function getVenvPythonExe(){return executable}\nexport function getVenvDir(){return ${literal(path.join(profile, 'dependencies/venv'))}}\nexport const getEmbeddedPythonDir=forbidden,getEmbeddedPythonExe=forbidden,ensureSslPatch=forbidden,markSetupDone=forbidden,runFullSetup=forbidden\nexport function getCertifiBundle(){return undefined}\n`)
  let bridge = fs.readFileSync(path.join(options.repo, 'electron/main/python-bridge.ts'), 'utf8').replaceAll('\r\n', '\n')
  bridge = replaceFunction(bridge, 'private async killProcessOnPort(): Promise<void>', `    await new Promise<void>((resolve,reject)=>{\n      const server=require('net').createServer();\n      server.once('error',(error:Error)=>reject(new Error('Preview requires unused 127.0.0.1:8765: '+error.message)));\n      server.listen({port:API_PORT,host:API_HOST,exclusive:true},()=>server.close((error?:Error)=>error?reject(error):resolve()));\n    })`)
  bridge = replaceFunction(bridge, 'private resolveApiDir(): string', `    return ${literal(apiDir)}`)
  bridge = replaceFunction(bridge, 'private resolvePythonExecutable(): string', `    const executable=${literal(options.apiPython)};\n    if(!existsSync(executable))throw Error('Configured existing preview API Python is missing');\n    return executable`)
  bridge = replaceFunction(bridge, 'private async waitUntilReady(maxRetries = 180, delayMs = 500): Promise<void>', `    const nonce=process.env.MODLY_AMD_PREVIEW_NONCE;\n    if(!nonce)throw Error('Preview launcher nonce is missing');\n    for(let i=0;i<maxRetries;i++){\n      if(!this.process)throw Error('Preview API exited during startup');\n      try{const reply=await axios.get(API_BASE_URL+'/health',{timeout:2000});\n        if(reply.data?.status!=='ok'||reply.data?.preview_nonce!==nonce||reply.data?.generation_enabled !== false)throw Error('Preview API identity mismatch');\n        this.ready=true;return;\n      }catch{await new Promise(resolve=>setTimeout(resolve,delayMs));}\n    }\n    throw Error('Preview API did not start with the launcher identity')`)
  bridge = replaceFunction(bridge, 'async stop(): Promise<void>', `    if(!this.process)return;\n    const proc=this.process;this.process=null;this.ready=false;this.intentionalStop=true;\n    await new Promise<void>(resolve=>{\n      if(proc.exitCode!==null||proc.signalCode!==null){resolve();return;}\n      proc.once('exit',()=>resolve());\n      if(!proc.kill()){resolve();return;}\n      const deadline=setTimeout(()=>{try{proc.kill('SIGKILL')}catch{}resolve()},3000);deadline.unref();\n    })`)
  if (!bridge.includes("'main:app'")) throw Error('Preview bridge entry signature changed')
  bridge = bridge.replace("'main:app'", "'preview_api:app'")
  result.set('python-bridge', bridge)
  let main = fs.readFileSync(path.join(options.repo, 'electron/main/index.ts'), 'utf8').replaceAll('\r\n', '\n')
  main = main.replace("app.setName('Modly')", "app.setName('Noncanonical AMD Image Test Preview')")
  main = main.replace("width: 1280,", "title: 'Noncanonical AMD Image Test Preview',\n    width: 1280,")
  main = main.replace("    shell.openExternal(details.url)", "    // External navigation is disabled in this private CPU preview.")
  main = main.replace("electronApp.setAppUserModelId('com.modly.app')", "electronApp.setAppUserModelId('com.modly.amd-image-test-preview')")
  // This compact, inert label preserves the exact qualified renderer/preload/CSS bytes.
  const banner = `(()=>{document.title='Noncanonical AMD Image Test Preview';const el=document.createElement('div');el.id='amd-image-test-preview-brand';el.textContent='Noncanonical AMD Image Test Preview · CPU image validation only';el.style.cssText='position:fixed;right:12px;bottom:5px;z-index:2147483646;font:10px sans-serif;color:#f0cc85;background:#171719dd;padding:2px 6px;border-radius:3px;pointer-events:none';document.body.appendChild(el)})()`
  main = main.replace("  mainWindow.on('ready-to-show'", `  mainWindow.webContents.on('dom-ready',()=>{mainWindow?.webContents.executeJavaScript(${JSON.stringify(banner)}).catch(()=>{})})\n\n  mainWindow.on('ready-to-show'`)
  result.set('index', main)
  return result
}

export function renderBootstrap(options) {
  const template = fs.readFileSync(path.join(HERE, 'preview-bootstrap.cjs'), 'utf8')
  return template.replace('__MODLY_PREVIEW_OPTIONS__', JSON.stringify({
    profile: options.profile ?? path.join(options.destination, 'profile'), cpuPython: options.cpuPython }))
}

export function treeHashes(root) {
  const files = {}
  function visit(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true }).sort((a,b) => a.name.localeCompare(b.name, 'en'))) {
      const filename = path.join(directory, entry.name)
      const stat = fs.lstatSync(filename)
      if (stat.isSymbolicLink()) throw Error('Runtime tree must contain regular files/directories only: ' + filename)
      if (stat.isDirectory()) visit(filename)
      else if (stat.isFile()) files[path.relative(root, filename).replaceAll('\\', '/')] = fileSha(filename)
      else throw Error('Unexpected runtime tree entry: ' + filename)
    }
  }
  visit(root)
  return files
}

export function treeDigest(files) {
  return sha(Object.keys(files).sort().map(name => `${name}\0${files[name]}\n`).join(''))
}

function runtimeProbe(executable, script, kind, profile) {
  const env = { ...process.env }
  for (const key of ['PYTHONHOME','PYTHONPATH','PYTHONSTARTUP','PYTHONUSERBASE','VIRTUAL_ENV','CONDA_PREFIX','PIP_USER','PIP_TARGET','PIP_PREFIX']) delete env[key]
  Object.assign(env, { PYTHONDONTWRITEBYTECODE:'1', PYTHONNOUSERSITE:'1', HF_HUB_OFFLINE:'1', TRANSFORMERS_OFFLINE:'1',
    CUDA_VISIBLE_DEVICES:'', HIP_VISIBLE_DEVICES:'', ROCR_VISIBLE_DEVICES:'', TEMP:path.join(profile,'temp'), TMP:path.join(profile,'temp'),
    HF_HOME:path.join(profile,'cache/huggingface'), TORCH_HOME:path.join(profile,'cache/torch'), XDG_CACHE_HOME:path.join(profile,'cache') })
  const result = spawnSync(executable, ['-B','-I',script,kind], { env, encoding:'utf8', timeout:10000, maxBuffer:16384, windowsHide:true })
  if (result.status !== 0 || result.error) throw Error(`Existing ${kind} runtime probe failed: ${result.error?.message ?? result.stderr}`)
  return JSON.parse(result.stdout)
}

export function windowsLauncher(apiPython) {
  return `@echo off\r\nsetlocal\r\nif not "%~2"=="" goto bad_args\r\nif "%~1"=="" goto normal\r\nif "%~1"=="--smoke-test" goto smoke\r\ngoto bad_args\r\n:normal\r\n"${apiPython}" -B -I -S "%~dp0launch_preview.py" --config "%~dp0preview.json"\r\ngoto finished\r\n:smoke\r\n"${apiPython}" -B -I -S "%~dp0launch_preview.py" --config "%~dp0preview.json" --test-timeout 180\r\ngoto finished\r\n:bad_args\r\necho Only the fixed developer option --smoke-test is supported.\r\nexit /b 2\r\n:finished\r\nif errorlevel 1 (\r\n  echo.\r\n  echo Preview could not start or close normally. Read the error above.\r\n  pause\r\n  exit /b 1\r\n)\r\nexit /b 0\r\n`
}

export function admitRuntimeProbes(api, cpu) {
  if (api.components?.Pillow_available !== false) throw Error('Qualified API must continue using the explicit CPU decoder; API-local Pillow changed')
  if (cpu.version !== '3.12.10' || cpu.components?.Pillow !== '12.3.0' || cpu.components?.webp !== true) {
    throw Error('Qualified existing CPU image decoder capability/version changed')
  }
}

export async function buildPreview(options) {
  const plan = createPlan(options)
  if (path.resolve(plan.apiPython).toLowerCase() !== path.resolve(API_PYTHON).toLowerCase() ||
      path.resolve(plan.cpuPython).toLowerCase() !== path.resolve(CPU_PYTHON).toLowerCase()) {
    throw Error('Only the previously qualified existing API and CPU Python paths are accepted; no runtime substitution')
  }
  if (fileSha(plan.apiPython) !== QUALIFIED_API_SHA || fileSha(plan.cpuPython) !== QUALIFIED_CPU_SHA) {
    throw Error('Previously qualified existing Python binary identity changed')
  }
  const app = path.join(plan.destination, 'runtime/resources/app')
  const seedApp = path.join(plan.seedRuntime, 'resources/app')
  const needed = ['launch_preview.py','runtime_probe.py','preview_api.py']
  for (const name of needed) if (!fs.statSync(path.join(HERE,name)).isFile()) throw Error('Preview source missing: ' + name)
  const sourceCommit = spawnSync('git', ['rev-parse','HEAD'], {cwd:plan.repo,encoding:'utf8',windowsHide:true})
  if (sourceCommit.status !== 0 || sourceCommit.stdout.trim() !== BASE) throw Error('Qualified source commit changed; review and select a new preview before building')
  const seedFiles = treeHashes(plan.seedRuntime)
  const require = createRequire(path.join(seedApp, 'package.json'))
  const esbuild = require('esbuild')
  if (esbuild.version !== '0.28.2') throw Error('Qualified existing esbuild version changed')
  // mkdir without recursive is the exclusive build claim. A failed build is preserved for diagnosis.
  fs.mkdirSync(plan.destination)
  fs.cpSync(plan.seedRuntime, path.join(plan.destination,'runtime'), { recursive:true, force:false, errorOnExist:true })
  for (const directory of ['temp','models','workspace','workflows','extensions','dependencies','agent','builtin-extensions','session','logs','cache']) {
    fs.mkdirSync(path.join(plan.profile,directory),{recursive:true})
  }
  const settings = Object.fromEntries(['models','workspace','workflows','extensions','dependencies','agent'].map(name=>[name+'Dir',path.join(plan.profile,name)]))
  put(path.join(plan.profile,'settings.json'), JSON.stringify(settings,null,2)+'\n')
  fs.mkdirSync(plan.apiDir)
  for (const name of needed) fs.copyFileSync(path.join(HERE,name), path.join(plan.destination,name === 'preview_api.py' ? 'api/preview_api.py' : name))
  fs.copyFileSync(path.join(plan.repo,'scripts/compatibility/amd_forward_owned_job.py'),path.join(plan.destination,'owned_job.py'))
  for (const name of ['routers/amd_image_test.py','services/amd_image_test.py','services/amd_stage_contract.py','services/amd_completed_result.py','services/amd_runtime.py']) {
    put(path.join(plan.apiDir,name),fs.readFileSync(path.join(plan.repo,'api',name)))
  }
  for (const name of ['routers/__init__.py','services/__init__.py']) put(path.join(plan.apiDir,name),'')
  const sources = guardedSources(plan)
  const guards = []
  for (const [name, guarded] of sources) {
    const source = path.join(plan.repo,'electron/main',name+'.ts')
    put(path.join(plan.destination,'guarded-source',name+'.ts'),guarded)
    guards.push({ source:'electron/main/'+name+'.ts', original_sha256:fileSha(source), guarded_sha256:sha(guarded) })
  }
  const guardPlugin = { name:'isolated-preview-main-guards', setup(build) {
    build.onLoad({filter:/electron[\\/]main[\\/][^\\/]+\.ts$/},args=>{
      const source = sources.get(path.basename(args.path,'.ts'))
      return source === undefined ? undefined : { contents:source,loader:'ts',resolveDir:path.dirname(args.path) }
    })
  }}
  const main = await esbuild.build({ absWorkingDir:plan.repo,bundle:true,logLevel:'warning',metafile:true,
    entryPoints:['electron/main/index.ts'],outfile:path.join(app,'out/main/index.js'),platform:'node',format:'cjs',packages:'external',
    nodePaths:[path.join(seedApp,'node_modules')],tsconfig:path.join(plan.repo,'tsconfig.json'),plugins:[guardPlugin] })
  put(path.join(plan.destination,'metafile-main.json'),JSON.stringify(main.metafile,null,2)+'\n')
  put(path.join(app,'out/main/preview-bootstrap.cjs'),renderBootstrap(plan))
  fs.unlinkSync(path.join(app,'out/main/smoke-bootstrap.cjs'))
  put(path.join(app,'package.json'),JSON.stringify({name:'modly-noncanonical-amd-image-test-preview',version:'0.4.3',main:'out/main/preview-bootstrap.cjs'},null,2)+'\n')
  put(path.join(plan.destination,'Launch AMD Image Test.cmd'),windowsLauncher(plan.apiPython))
  const apiProbe = runtimeProbe(plan.apiPython,path.join(plan.destination,'runtime_probe.py'),'api',plan.profile)
  const cpuProbe = runtimeProbe(plan.cpuPython,path.join(plan.destination,'runtime_probe.py'),'cpu',plan.profile)
  admitRuntimeProbes(apiProbe,cpuProbe)
  const runtimeFiles = treeHashes(path.join(plan.destination,'runtime'))
  const immutable = ['launch_preview.py','runtime_probe.py','owned_job.py','Launch AMD Image Test.cmd','profile/settings.json']
  for (const name of Object.keys(treeHashes(plan.apiDir))) immutable.push('api/'+name)
  // Full runtime identity is separately authenticated by runtime_tree_sha256.
  for (const name of Object.keys(runtimeFiles)) {
    if (!name.includes('/node_modules/')) immutable.push('runtime/'+name)
  }
  if (immutable.length > 256) throw Error('Critical-file closure exceeded launcher manifest bound; review required')
  const files = Object.fromEntries(immutable.sort().map(name=>[name,fileSha(path.join(plan.destination,name))]))
  const config = {schema:SCHEMA,source_commit:BASE,root:plan.destination,executable:path.join(plan.destination,'runtime/Modly.exe'),
    profile:plan.profile,api_dir:plan.apiDir,api_python:plan.apiPython,cpu_python:plan.cpuPython,api_probe:apiProbe,cpu_probe:cpuProbe,
    runtime_tree_sha256:treeDigest(runtimeFiles),files}
  put(path.join(plan.destination,'preview.json'),JSON.stringify(config,null,2)+'\n')
  const preserved = Object.keys(seedFiles).filter(name=>name.startsWith('resources/app/out/renderer/')||name.startsWith('resources/app/out/preload/'))
  for (const name of preserved) if (seedFiles[name] !== runtimeFiles[name]) throw Error('Qualified renderer/preload/CSS was changed: '+name)
  const receipt = {schema:'modly.amd.image-test-preview-build.v1',classification:'BUILT_NOT_LAUNCHED',source_commit:BASE,
    kind:'Noncanonical existing-runtime preview; reused qualified renderer/preload/CSS; main-only esbuild',esbuild:esbuild.version,
    builder_sha256:fileSha(fileURLToPath(import.meta.url)),bootstrap_template_sha256:fileSha(path.join(HERE,'preview-bootstrap.cjs')),
    options,guards,qualified_api_python_sha256:QUALIFIED_API_SHA,qualified_cpu_python_sha256:QUALIFIED_CPU_SHA,
    qualified_runtime_provenance_sha256:fileSha(path.resolve(plan.seedRuntime,'../../smoke-host-jsx-inputs.json')),
    renderer_preload_files_preserved:preserved.length,seed_runtime_tree_sha256:treeDigest(seedFiles),
    runtime_tree_sha256:treeDigest(runtimeFiles),runtime_files_sha256:runtimeFiles,
    limitation:'No canonical Vite/Tailwind/TypeScript chain was available. Copied installed CSS does not regenerate new utility classes. No model, GPU, WSL, shell, install, update or generation execution is enabled.'}
  put(path.join(plan.destination,'build-receipt.json'),JSON.stringify(receipt,null,2)+'\n')
  return { classification:receipt.classification, destination:plan.destination, runtime_tree_sha256:config.runtime_tree_sha256,
    runtime_files:Object.keys(runtimeFiles).length, critical_files:immutable.length, renderer_preload_files_preserved:preserved.length }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw Error('Usage: node build_preview.mjs absolute-build-config.json')
  console.log(JSON.stringify(await buildPreview(JSON.parse(fs.readFileSync(process.argv[2],'utf8')))))
}
