const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, 'viewer_observation.js'), 'utf8');
function fixture({ wrongModelStore = false } = {}) {
  let tick = 0;
  const events = object => Object.assign(object, {
    listeners: new Map(),
    addEventListener(type, fn) { const a=this.listeners.get(type)||[];a.push(fn);this.listeners.set(type,a); },
    removeEventListener(type, fn) { this.listeners.set(type,(this.listeners.get(type)||[]).filter(v=>v!==fn)); },
    emit(type,event={}) { for(const fn of this.listeners.get(type)||[]) fn(event); }
  });
  const canvas = { tagName: 'CANVAS' }, input = { contains: e=>e===canvas };
  const camera = { type:'PerspectiveCamera', position:{x:0,y:1.5,z:4}, quaternion:{x:0,y:0,z:0,w:1}, zoom:1,
    projectionMatrix:{elements:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]},
    matrixWorldInverse:{elements:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]} };
  const controls=events({object:camera,domElement:input,target:{x:0,y:0,z:0},mouseButtons:{LEFT:0,MIDDLE:1,RIGHT:2},enabled:true});
  const geometry=events({uuid:'geometry',boundsTree:{},boundingBox:{min:{x:-.2,y:-.3,z:-.4},max:{x:.2,y:.3,z:.4}}});
  const material=events({uuid:'material'});
  const scene={uuid:'main-scene'}, mesh={isMesh:true,geometry,material,userData:{originalMaterial:material},
    matrixWorld:{elements:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}};
  const model={uuid:'model',parent:scene,position:{x:0,y:0,z:0},quaternion:{x:0,y:0,z:0,w:1},scale:{x:1,y:1,z:1},traverse(fn){fn(this);fn(mesh);}};
  const calls=[];
  const original=function(...args){calls.push({receiver:this,args});if(args[2]==='throw')throw Error('original-error');return 'original-result';};
  const gl={domElement:canvas,render:original,info:{memory:{geometries:1,textures:0},programs:[],render:{frame:0}}};
  let frameCallback = null, subscription = null;
  const state={gl,scene,camera,controls,events:{connected:input},internal:{subscribe(ref,priority,rootStore){
    subscription={ref,priority,rootStore};frameCallback=ref.current;return()=>{frameCallback=null;};
  }}};
  const store={getState:()=>state};
  model.__r3f={root:wrongModelStore?{getState:()=>state}:store};
  function CanvasCapture(){} function SceneMeshModel(){} function Provider(){}
  const helper={type:Provider,memoizedProps:{store:{getState:()=>({gl,camera:{type:'CubeCamera'},scene:{},events:{connected:input}})}}};
  const provider={type:Provider,memoizedProps:{store}};
  const capture={type:CanvasCapture,memoizedProps:{domRef:{current:canvas}}};
  const primitive={type:'primitive',memoizedProps:{object:controls},stateNode:controls};
  const defaults={type:{render(){}},memoizedProps:{makeDefault:true},child:primitive};
  const modelFiber={type:SceneMeshModel,memoizedProps:{scene:model}};
  helper.sibling=provider;provider.sibling=capture;capture.sibling=defaults;defaults.sibling=modelFiber;
  const roots=new Set([{current:{child:helper}}]);
  const window=events({});
  const context={WeakRef,window,performance:{now:()=>++tick},roots};
  const install=()=>vm.runInNewContext('('+source+')(roots)',context);
  return { install,gl,original,calls,scene,camera,controls,window,input,model,modelFiber,geometry,material,roots,
    frame(){frameCallback?.();},getSubscription:()=>subscription,isSubscribed:()=>!!frameCallback,store };
}
test('selects validated main camera despite preceding helper camera; records conservative framing',()=>{
  const f=fixture(),api=f.install(),s=api.trackModel('master');
  assert.equal(s.cameraType,'PerspectiveCamera');
  assert.equal(s.currentModels[0].framing.allCornersInside,true);
  assert.ok(Math.abs(s.currentModels[0].framing.viewportHeightFraction-.3)<1e-10);
  assert.deepEqual(JSON.parse(JSON.stringify(s.currentModels[0].transform)),{position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]});
  assert.equal(s.resources.length,2); // original=current material must deduplicate
  assert.equal(s.controls.keyboardListenerConnected,false);
  api.stop();
});
test('rejects scene associated with another store before wrapping render',()=>{
  const f=fixture({wrongModelStore:true});
  assert.throws(f.install,/one authenticated main viewer/);
  assert.equal(f.gl.render,f.original);
});
test('render wrapper preserves receiver args return throw, ignores helper renders, and restores',()=>{
  const f=fixture(),api=f.install();
  assert.equal(f.gl.render(f.scene,f.camera,'extra'),'original-result');
  assert.equal(f.calls[0].receiver,f.gl);
  assert.deepEqual(f.calls[0].args,[f.scene,f.camera,'extra']);
  f.gl.render({},f.camera);f.gl.render(f.scene,{});
  assert.equal(api.snapshot().mainRenderCalls,1);
  assert.throws(()=>f.gl.render(f.scene,f.camera,'throw'),/original-error/);
  assert.equal(api.snapshot().mainRenderCalls,1);
  api.stop();assert.equal(f.gl.render,f.original);api.stop();
});
test('requires trusted input then camera pose change then actual successful main render',()=>{
  const f=fixture(),api=f.install();api.beginWindow('orbit');
  const input={isTrusted:false,type:'pointermove',buttons:1,button:-1,composedPath:()=>[f.input]};
  f.window.emit('pointermove',input);f.controls.emit('change');f.gl.render(f.scene,f.camera);
  input.isTrusted=true;f.window.emit('pointermove',input);
  f.camera.position.x+=1;f.controls.emit('change');
  f.gl.render({},f.camera); // environment helper is not evidence
  f.gl.render(f.scene,f.camera);
  const result=api.endWindow();assert.equal(result.inputs.length,1);
  const row=result.inputs[0];assert.ok(row.changeAt>row.at);assert.ok(row.renderAt>=row.changeAt);
  assert.ok(row.renderReturnAt>=row.renderAt);assert.equal(result.mainRenderCalls,2);
  api.stop();
});
test('weak resource ledger sees unmount disposal and BVH clear while main canvas stays alive',()=>{
  const f=fixture(),api=f.install();api.trackModel('master');
  f.geometry.emit('dispose');f.material.emit('dispose');f.geometry.boundsTree=null;
  f.model.parent=null;f.modelFiber.memoizedProps.scene=f.model;
  let s=api.snapshot();assert.equal(s.models[0].attachedToMainScene,false);
  assert.equal(s.resources.find(r=>r.kind==='geometry').disposals,1);
  assert.equal(s.resources.find(r=>r.kind==='geometry').bvhPresent,false);
  assert.equal(s.resources.find(r=>r.kind==='material').disposals,1);
  api.stop();assert.equal(f.geometry.listeners.get('dispose').length,0);
});
test('numbers actual R3F ticks, groups duplicate render passes, and unsubscribes on stop',()=>{
  const f=fixture(),api=f.install(),subscription=f.getSubscription();
  assert.equal(subscription.priority,0);assert.equal(subscription.rootStore,f.store);
  api.beginWindow('zoom');
  f.frame();f.gl.render(f.scene,f.camera);f.gl.render(f.scene,f.camera);
  f.frame();f.gl.render(f.scene,f.camera);f.gl.render(f.scene,f.camera);
  f.frame();f.gl.render(f.scene,f.camera);
  const result=api.endWindow();
  assert.equal(result.frameTicks,3);assert.equal(result.renderedFrameTicks,3);
  assert.equal(result.frameIntervals.length,2);assert.equal(result.renderedFrameIntervals.length,2);
  assert.equal(result.mainRenderCalls,5);assert.equal(result.mainRenderIntervals.length,4);
  api.stop();assert.equal(f.isSubscribed(),false);
});
