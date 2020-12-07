import yaml
from collections import defaultdict

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d

def Config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        dictionary = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dictToObj(dictionary)
    return cfg

if __name__ == "__main__":
    cfg = Config()
    print(cfg.optimizer.lr)

