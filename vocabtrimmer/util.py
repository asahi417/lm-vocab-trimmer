import shutil
import os


DEFAULT_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/vocabtrimmer"


def safe_rmtree(path): shutil.rmtree(path) if os.path.exists(path) else None


def pretty(num): return "{:,}".format(num)


def get_cache_dir(root_dir):
    _id = 0
    while True:
        path = f"{root_dir}.{_id}"
        if not os.path.exists(path):
            break
        _id += 1
    return path
