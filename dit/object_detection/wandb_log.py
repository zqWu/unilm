import datetime
import os
import pathlib
import re
import signal
import threading
import time
from typing import Optional

import wandb


class AppendOnlyTextFileMonitor(threading.Thread):
    """watch file. condition:
    1. text file
    2. append only
    so it can read line by line
    """

    def __init__(self, abs_path, line_processor):
        super().__init__(daemon=False)
        self.last_line_num = 0
        self.file = open(abs_path, 'rt')
        self.line_processor = line_processor
        self.total_line = 0
        self.valid_line = 0

    def run(self):
        while True:
            for line_txt in self.file:
                self.total_line += 1
                # print(line_txt.rstrip())
                is_valid = self.line_processor(line_txt)
                if is_valid:
                    self.valid_line += 1
                if self.total_line % 100 == 99:
                    print(f'total_line={self.total_line}, valid_line={self.valid_line}')


def init_wandb():
    wandb.login(key='xxxx')
    run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(
        project="dit",
        name=f"run-{run_name}",
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "doclaynet",
            "epochs": 10,
        }
    )


# This function forcibly kills the remaining wandb process.
def force_finish_wandb():
    with open(os.path.join(os.path.dirname(__file__), './wandb/latest-run/logs/debug-internal.log'), 'r') as f:
        last_line = f.readlines()[-1]
    match = re.search(r'(HandlerThread:|SenderThread:)\s*(\d+)', last_line)
    if match:
        pid = int(match.group(2))
        print(f'wandb pid: {pid}')
    else:
        print('Cannot find wandb process-id.')
        return

    try:
        time.sleep(3)
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} killed successfully.")
    except OSError:
        print(f"Failed to kill process with PID {pid}.")


# Start wandb.finish() and execute force_finish_wandb() after 60 seconds.
def try_finish_wandb():
    threading.Timer(10, force_finish_wandb).start()
    wandb.finish()


def extract_pattern_from_line(line: str) -> Optional[dict]:
    """从 单行文本中提取 信息"""
    name_and_pattern_cascade = {
        "total_loss": r'total_loss: (\d+\.\d+)',
        "loss_box_reg_stage0": r'loss_box_reg_stage0: (\d+\.\d+)',
        "loss_cls_stage0": r'loss_cls_stage0: (\d+\.\d+)',
        "loss_box_reg_stage1": r'loss_box_reg_stage1: (\d+\.\d+)',
        "loss_cls_stage1": r'loss_cls_stage1: (\d+\.\d+)',
        "loss_box_reg_stage2": r'loss_box_reg_stage2: (\d+\.\d+)',
        "loss_cls_stage2": r'loss_cls_stage2: (\d+\.\d+)',
    }

    name_and_pattern_maskrcnn = {
        "total_loss": r'total_loss: (\d+\.\d+)',
        "loss_cls": r'loss_cls: (\d+\.\d+)',
        "loss_box_reg": r'loss_box_reg: (\d+\.\d+)',
        "loss_mask": r'loss_mask: (\d+\.\d+)',
        "loss_rpn_cls": r'loss_rpn_cls: (\d+\.\d+)',
        "loss_rpn_loc": r'loss_rpn_loc: (\d+\.\d+)',
    }
    name_and_pattern = name_and_pattern_cascade

    if not ("d2.utils.events" in line and "total_loss" in line):
        return None

    line = line.strip()
    loss_info = {}
    for name, pattern in name_and_pattern.items():
        match_case = re.search(pattern, line)
        if match_case:
            loss_info[name] = float(match_case.group(1))
        else:
            print(f"not match {name}")
    return loss_info


class LogFileLineProcessor:
    def __init__(self, logger, sleep_interval):
        self.logger = logger
        self.sleep_interval = sleep_interval

    def process_line(self, line: str):
        """处理单行文本"""
        loss_info = extract_pattern_from_line(line)
        if loss_info:
            self.logger.log(loss_info)
            if self.sleep_interval > 0:
                time.sleep(self.sleep_interval)
            return True
        return False


def test_no_monitor():
    init_wandb()
    file_path = "train_maskrcnn.log"
    processor = LogFileLineProcessor(wandb, sleep_interval=-1)

    with open(file_path, "rt") as file:
        count = 0
        for line in file:
            is_valid = processor.process_line(line)
            if is_valid:
                count += 1

    wandb.finish()
    # try_finish_wandb()


def monitor_log_file():
    init_wandb()
    curr_dir = pathlib.Path(__file__).resolve().parent
    abs_path = os.path.join(curr_dir, 'train.log')
    # abs_path = os.path.join(curr_dir, 'train_maskrcnn.log')
    processor = LogFileLineProcessor(wandb, sleep_interval=-1)
    monitor = AppendOnlyTextFileMonitor(abs_path=abs_path, line_processor=processor.process_line)
    monitor.start()


# test_no_monitor()
monitor_log_file()
