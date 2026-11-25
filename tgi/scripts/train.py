# TGI 世界模型训练的命令行入口。训练阶段的具体调度在 training/trainer.py 中。

"""Train a Terrain Grounded Imagination policy for the Unitree G1 (see README.md for the options).

    python scripts/train.py --log_dir logs/<run_name> [options]
"""

from tgi.training.config import parse_args


# 先解析参数，再加载训练器、创建仿真环境并启动训练循环。
def main():
    cfg = parse_args()  # before torch and Isaac Lab are imported
    from tgi.training.trainer import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    main()
