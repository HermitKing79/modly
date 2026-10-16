"""Compare selected runtime functions with code compiled from authenticated bytes.

Compilation executes no source statement, import, class body or constructor.
This is a bounded identity check for the reviewed original methods, not a JIT.
"""
from hashlib import sha256
from pathlib import Path
from types import CodeType, FunctionType

from .admission import require


def _constant(value):
    if isinstance(value, CodeType):
        return ('code', _signature(value))
    if type(value) is tuple:
        return ('tuple', tuple(_constant(item) for item in value))
    if type(value) is frozenset:
        return ('frozenset', frozenset(_constant(item) for item in value))
    return (type(value), value)


def _signature(code):
    # Include nested function bodies and exception handling, not source-location
    # metadata which a foreign function can copy with code.replace().
    return (code.co_name, code.co_qualname, code.co_argcount, code.co_posonlyargcount,
            code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags,
            code.co_code, tuple(_constant(value) for value in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            code.co_exceptiontable)


def qualify_function(function, source_sha256, *qualified_names, defaults=None):
    """Authenticate actual executable code, including its nested functions."""
    require(type(function) is FunctionType and qualified_names,
            'original Python function required for source-code qualification')
    blob = Path(function.__code__.co_filename).read_bytes()
    require(sha256(blob).hexdigest() == source_sha256, 'runtime function source identity differs')
    expected = compile(blob, function.__code__.co_filename, 'exec', dont_inherit=True)
    for name in qualified_names:
        candidates = [value for value in expected.co_consts
                      if isinstance(value, CodeType) and value.co_name == name]
        require(candidates, 'selected original source function/class is missing')
        # SparseTensor declares @overload __init__ stubs before its real final
        # definition. Python binds that last definition in the original class.
        expected = candidates[-1]
    require(_signature(function.__code__) == _signature(expected),
            'runtime executable code differs from authenticated original source')
    require(function.__defaults__ == defaults and function.__kwdefaults__ in (None, {}),
            'runtime function defaults differ from authenticated original contract')
    return function
