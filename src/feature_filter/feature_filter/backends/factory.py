import importlib

def instance(params, client, server, size):
    module = importlib.import_module("feature_filter.backends")
    backend_class = getattr(module, params["backend"]+"Backend")
    backend = backend_class(client, server, size, params)
    return backend