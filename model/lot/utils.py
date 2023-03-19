from ctypes import cast, py_object


def _get_id(*argv):
    _id = []
    for arg in argv:
        _id.append(id(arg))
    return tuple(_id)

def _get_obj(*argv):
    _obj = []
    for arg in argv:
        _method = cast(arg, py_object).value
        if callable(_method) and '__' not in str(_method):
            _obj.append(_method)
    return tuple(_obj)

def _get_input(*argv, InputType):
    _input = []
    for arg in argv:
        _input.append(InputType)
    return tuple(_input)

def _get_return(*argv):
    _return = []
    for arg in argv:
        _return.append(arg.__annotations__['return'])
    return tuple(_return)