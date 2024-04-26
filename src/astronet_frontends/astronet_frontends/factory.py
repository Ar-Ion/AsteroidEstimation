import importlib

def instance(params, mode, size):
    module = importlib.import_module("astronet_frontends")
    frontend_class = getattr(module, params["type"])
    frontend = frontend_class(params["path"], mode, size)
    return frontend