import logging


def setup_logging(log_level="INFO", log_path=None, no_console_log=False):
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)

    if no_console_log:
        logging.getLogger().handlers = [
            h
            for h in logging.getLogger().handlers
            if not isinstance(h, logging.StreamHandler)
        ]
