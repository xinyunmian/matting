
import sys
sys.path.append('.')
import argparse
from core.engine import joint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/human.yaml', help='config file(*.yaml)')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--phase', type=str, default='seg', help='train or eval')
    args = parser.parse_args()

    engine = joint.EngineJHMP(args.config)
    engine.run(mode=args.mode)