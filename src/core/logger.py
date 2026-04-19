# src/core/logger.py
import logging
import sys
from src.core.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    获取统一配置的 logger
    :param name: 通常传 __name__
    :return: 配置好的 logger 对象
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # 控制台输出 handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # 日志格式：时间 - 名称 - 级别 - 消息
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # 防止向上传播到 root logger

    return logger