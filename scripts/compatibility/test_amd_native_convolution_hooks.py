"""Run only inside the owned GPU-blocked AMD CPU worker, using installed knobs."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch

CHAIN_NAMES=('launch_enter_hook','launch_exit_hook','kernel_load_start_hook','kernel_load_end_hook','kernel_unload_hook')

def run_constructor_suite(torch,private,meta,source):
    import triton
    from triton import knobs
    import triton.compiler.compiler as compiler
    import triton.runtime.build as build
    import triton.backends.amd.driver as driver
    from triton.backends.amd.compiler import HIPBackend
    from amd_forward_diagnostics import Trace
    from amd_native_convolution_policy import NativePolicy,validate_hook_state
    traces=[];callbacks=[]

    @contextlib.contextmanager
    def constructed():
        policy=NativePolicy(torch,private,meta,source,Trace(traces.append))
        restores=policy.restores.copy()
        try:yield policy
        finally:
            policy.close()
            for obj,name,original in restores:
                if getattr(obj,name) is not original:raise AssertionError('Production constructor patch not restored: '+name)

    def reject():
        try:
            with constructed():pass
        except ValueError:return
        raise AssertionError('Foreign/nonempty hook or altered policy accepted')

    class InstalledConstructorTests(unittest.TestCase):
        def test_complete_real_default_constructor(self):
            with constructed() as policy:
                required={(HIPBackend,'add_stages'),(compiler.ASTSource,'make_ir'),(driver,'compile_module_from_src'),
                    (build,'_build'),(build,'_load_module_from_path'),(build,'compile_module_from_src'),(build,'compile_module_from_file'),
                    (compiler,'get_cache_manager'),(compiler,'compile'),(triton,'compile'),(triton.compiler,'compile'),
                    (compiler.CompiledKernel,'_init_handles'),(driver.HIPLauncher,'__call__'),
                    (triton.runtime.JITFunction,'run'),(triton.runtime.JITFunction,'warmup'),(triton.runtime.Autotuner,'run'),(os,'mkdir')}
                self.assertEqual({(obj,name) for obj,name,_ in policy.restores},required)
                self.assertTrue(all(getattr(obj,name) is not original for obj,name,original in policy.restores))
                self.assertEqual(policy.permit.phase,'cpu')
                self.assertTrue(all(v==0 for v in policy.permit.counts.values()))
                self.assertEqual(policy.helper_calls,0);self.assertEqual(policy.lookups,[])
                self.assertEqual(policy.forbidden_calls,[])
        def test_supported_explicit_none(self):
            with contextlib.ExitStack() as stack:
                for name in CHAIN_NAMES:stack.enter_context(patch.object(knobs.runtime,name,None))
                with constructed():pass
        def test_nonempty_default_chains_rejected_without_calling_callbacks(self):
            for name in CHAIN_NAMES:
                chain=getattr(knobs.runtime,name)
                with patch.object(chain,'calls',[lambda:callbacks.append(name)]):reject()
            self.assertEqual(callbacks,[])
        def test_foreign_hook_objects_rejected(self):
            for name in CHAIN_NAMES:
                with patch.object(knobs.runtime,name,object()):reject()
        def test_foreign_empty_chain_instances_rejected(self):
            for name in CHAIN_NAMES:
                with patch.object(knobs.runtime,name,knobs.HookChain()):reject()
        def test_hook_subclasses_rejected(self):
            class Foreign(knobs.HookChain):pass
            for name in CHAIN_NAMES:
                with patch.object(knobs.runtime,name,Foreign()):reject()
        def test_class_default_replacements_rejected(self):
            for name in CHAIN_NAMES:
                replacement=knobs.HookChain(reversed=getattr(knobs.runtime,name).reversed)
                with patch.object(knobs.runtime_knobs,name,replacement):reject()
        def test_malformed_callback_containers_rejected(self):
            for name in CHAIN_NAMES:
                for calls in ((),None,{},0):
                    with patch.object(getattr(knobs.runtime,name),'calls',calls):reject()
        def test_changed_reverse_flags_rejected(self):
            for name in CHAIN_NAMES:
                chain=getattr(knobs.runtime,name)
                for reverse in (not chain.reversed, int(chain.reversed)):
                    with patch.object(chain,'reversed',reverse):reject()
        def test_unexpected_chain_attributes_rejected(self):
            for name in CHAIN_NAMES:
                with patch.object(getattr(knobs.runtime,name),'unexpected_callback',lambda:callbacks.append(name),create=True):reject()
            self.assertEqual(callbacks,[])
        def test_none_hooks_reject_callbacks(self):
            for obj,name in ((knobs.runtime,'add_stages_inspection_hook'),(knobs.runtime,'jit_cache_hook'),(knobs.runtime,'jit_post_compile_hook'),(knobs.compilation,'listener')):
                with patch.object(obj,name,lambda:callbacks.append(name)):reject()
            self.assertEqual(callbacks,[])
        def test_changed_compilation_policies_rejected(self):
            for name in ('always_compile','override','dump_ir','store_binary_only'):
                # Installed env descriptors also update process environment.
                # Their own scope restores both attributes and environment.
                with knobs.compilation.scope(),patch.object(knobs.compilation,name,True):reject()
            with patch.object(knobs.cache,'manager_class',object()):reject()
        def test_complete_constructor_preserves_deny_boundaries(self):
            with constructed() as policy:
                for fn in (lambda:compiler.get_cache_manager('missing'),lambda:build._build(),lambda:build._load_module_from_path(),
                           lambda:driver.compile_module_from_src(),lambda:HIPBackend.add_stages(),
                           lambda:compiler.CompiledKernel._init_handles(None),lambda:triton.runtime.JITFunction.run(None,grid=None,warmup=False)):
                    with self.assertRaises(RuntimeError):fn()
                self.assertTrue(all(v==0 for v in policy.permit.counts.values()))
                self.assertEqual(policy.lookups,[]);self.assertEqual(policy.helper_calls,0)

    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(InstalledConstructorTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    return dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        complete_constructor_passed=not any('test_complete_real_default_constructor' in str(case) for case,_ in result.errors+result.failures),
        callbacks_executed=len(callbacks),gpu_initialized=torch.cuda.is_initialized(),traces=traces,hook_state=validate_hook_state(knobs))
