import json

from utils import Log


def cache_obj(cache_path, obj, encoder):
    Log.d("cache_obj: cache_path=" + cache_path)
    json_str = json.dumps(obj, default=encoder)
    cache = open(cache_path, "w")
    cache.write(json_str)
    cache.close()


def get_cache_obj(cache_path, decoder):
    Log.d("cache_obj: cache_path=" + cache_path)
    cache = None
    try:
        cache = open(cache_path, "r")
        json_str = cache.read()
        return json.loads(json_str, object_hook=decoder)
    except:
        return None
    finally:
        if cache is not None:
            cache.close()
