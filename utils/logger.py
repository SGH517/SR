import logging
import os

def setup_logger(log_dir, log_file_name="log.txt", level=logging.INFO):
    """
    设置日志记录器，支持控制台和文件日志。

    参数：
        log_dir (str): 日志文件保存的目录。
        log_file_name (str): 日志文件的名称，默认 "log.txt"。
        level (int): 日志记录级别，默认 logging.INFO。

    返回：
        logger (logging.Logger): 配置好的日志记录器。
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)

    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 创建文件日志处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 添加处理器到日志记录器
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def set_logger_level(logger, level):
    """
    设置日志记录器的日志级别。

    参数：
        logger (logging.Logger): 日志记录器实例。
        level (int): 日志级别，例如 logging.DEBUG。
    """
    for handler in logger.handlers:
        handler.setLevel(level)
    logger.setLevel(level)

# 示例用法
if __name__ == "__main__":
    logger = setup_logger("logs", "example_log.txt")
    logger.info("这是一个信息日志。")
    logger.error("这是一个错误日志。")