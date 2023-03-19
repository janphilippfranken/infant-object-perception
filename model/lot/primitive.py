from utils import _get_id, _get_obj

class Primitive():
    def __init__(self, *argv):
        self._ids = _get_id(*argv)
        self._methods = _get_obj(*self._ids)
        self._inputs = {k: [] for k in [method.__annotations__['a'] for method in self._methods]}
        self._returns = {k: [] for k in [method.__annotations__['return'] for method in self._methods]}
        for method in self._methods:
            self._inputs[method.__annotations__['a']].append(method)
            self._returns[method.__annotations__['return']].append(method)