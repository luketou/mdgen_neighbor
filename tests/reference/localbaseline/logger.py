"""
【檔案功能說明】
本檔案 (logger.py) 負責設定與提供 MDGen 專案統一的日誌記錄器 (Logger)。
它支援將訓練與推論過程中的訊息同時輸出到終端機 (Console) 以及日誌檔案 (log.out)，
並包含自訂的日誌等級與格式化設定。
"""

import logging, socket, os

model_dir = os.environ.get("MODEL_DIR", "./workdir/default")

class Rank(logging.Filter):
    """
    【類別說明】
    用於在分散式訓練 (Distributed Training) 環境下，為日誌紀錄注入全域與本地 Rank (進程編號) 資訊的過濾器。
    """
    def filter(self, record):
        """
        過濾並附加進程 Rank 資訊至日誌紀錄中。
        """
        record.global_rank = os.environ.get("GLOBAL_RANK", 0)
        record.local_rank = os.environ.get("LOCAL_RANK", 0)
        return True


def get_logger(name):
    """
    【函式說明】
    建立並回傳一個設定好的 Logger 實例。

    參數：
        name (str): Logger 的名稱，通常傳入 __name__。

    回傳：
        logger (logging.Logger): 同時連接了 Console (StreamHandler) 與檔案 (FileHandler) 的日誌物件。
    """
    logger = logging.Logger(name)
    # logger.addFilter(Rank())
    logger = logging.Logger(name)
    # logger.addFilter(Rank())
    level = {"crititical": 50, "error": 40, "warning": 30, "info": 20, "debug": 10}[
        os.environ.get("LOGGER_LEVEL", "info")
    ]
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    os.makedirs(model_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(model_dir, "log.out"))
    fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(f'%(asctime)s [{socket.gethostname()}:%(process)d:%(global_rank)s:%(local_rank)s]
    # [%(levelname)s] %(message)s') #  (%(name)s)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger