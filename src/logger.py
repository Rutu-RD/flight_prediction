import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "app",log_dir: Path | str = "logs",level: int = logging.INFO,) -> logging.Logger:
	# log_dir = Path(log_dir)
	# log_dir.mkdir(parents=True, exist_ok=True)

	# log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.propagate = False  # prevent duplicate logs

	# Prevent adding handlers multiple times
	if logger.handlers:
		return logger

	formatter = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	# File handler
	#file_handler = logging.FileHandler(log_file)
	#file_handler.setFormatter(formatter)

	# Console handler
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)

	# logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	return logger
