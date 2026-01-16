import contextlib
import logging
import os

logger = logging.getLogger(__name__)

try:
    import matlab.engine
except ImportError:
    logger.info("Matlab engine not available")
    matlab = None


class MatlabEngine:
    _eng = None

    @classmethod
    def get_engine(cls):
        if cls._eng is None:
            raise RuntimeError("MATLAB engine not started.")
        return cls._eng

    @classmethod
    def start_engine(cls):
        if cls._eng is None:
            if matlab:
                logger.info("Starting MATLAB engine...")
                cls._eng = matlab.engine.start_matlab()
                logger.info("MATLAB engine started.")
            else:
                raise ImportError("MATLAB engine for Python is not installed.")
        return cls._eng

    @classmethod
    def terminate_engine(cls):
        if cls._eng is not None:
            logger.info("Terminating MATLAB engine.")
            cls._eng.quit()
            cls._eng = None


@contextlib.contextmanager
def matlab_engine():
    """Context manager for MATLAB engine."""
    engine = None
    try:
        engine = MatlabEngine.start_engine()
        yield engine
    finally:
        MatlabEngine.terminate_engine()

