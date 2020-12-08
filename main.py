from config import Config
from model import train

if __name__ == '__main__':
    cfg = Config()
    train(cfg)