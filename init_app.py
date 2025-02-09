import os
import sys
from config import *


def initialize_app():
	# 在已有的初始化代码中添加
	# 创建日志文件夹
	os.makedirs(LOG_FOLDER, exist_ok=True)
	print(f"Created log folder: {LOG_FOLDER}")

	# 创建日志文件
	for log_file in [DEBUG_LOG_FILE, ERROR_LOG_FILE]:
		if not os.path.exists(log_file):
			open(log_file, 'a').close()
			print(f"Created log file: {log_file}")


if __name__ == "__main__":
	initialize_app()
