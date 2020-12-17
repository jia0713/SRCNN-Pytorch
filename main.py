from config import Config
from model import train, eval

if __name__ == '__main__':
    cfg = Config()
    if cfg.is_train:
        train(cfg)
    else:
        eval(cfg)