import datetime
import logging
import logging.handlers
import os
import multiprocessing


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


# --- Setup ---
def setup_logging(
    log_dir: str = os.path.abspath(".logs"),
    log_prefix: str = "",
    log_level_console: int = logging.INFO,
    log_level: int = logging.INFO,
):
    """Configures the application's logging."""
    # Only configure logging in the main process
    if multiprocessing.current_process().name != "MainProcess":
        return logging.getLogger()

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    if log_prefix == "":
        log_file = f"{timestamp}.log"
    else:
        log_file = f"{log_prefix}_{timestamp}.log"
    log_file = os.path.join(log_dir, log_file)

    print(f"Logging to {log_file}")

    # 1. Get the logger
    logger = logging.getLogger()
    # Set the lowest-level to be captured. DEBUG will capture everything.
    logger.setLevel(min(log_level_console, log_level))

    # 2. Create a formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # 3. Create and configure the console handler (for development)
    # This handler prints logs to the console (stdout).
    console_handler = logging.StreamHandler()
    # You can set a different level for the console. For example, INFO.
    # The level can be controlled by an environment variable for flexibility.
    console_handler.setLevel(log_level_console)
    console_handler.setFormatter(formatter)

    # 4. Create and configure the rotating file handler (for production/history)
    # This handler writes logs to a file, with automatic rotation.
    file_handler = logging.FileHandler(log_file)

    # The file handler should typically log everything (DEBUG and up).
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 5. Add handlers to the logger
    # Avoid adding handlers if they already exist (e.g., in interactive sessions)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent logs from propagating to the root logger
    # logger.propagate = False

    # 6. Log the successful configuration
    logger.debug("Logging has been configured successfully. Console level: %s", log_level_console)

    return logger
