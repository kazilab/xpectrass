import logging
import warnings
from contextlib import contextmanager

@contextmanager
def log_and_suppress_warnings(log_path: str, *, categories=(RuntimeWarning,), ignore_in_notebook=True):
    """
    Capture warnings during the context and write them to a log file.
    Optionally suppress them from notebook display.
    """
    logger = logging.getLogger("ftir_warnings")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if cell re-runs
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_path for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # record all warnings
        if ignore_in_notebook:
            for cat in categories:
                warnings.filterwarnings("ignore", category=cat)

        yield  # run the code

        # Write captured warnings to file
        for w in caught:
            if categories is None or any(issubclass(w.category, c) for c in categories):
                msg = (
                    f"{w.category.__name__}: {w.message} | "
                    f"{w.filename}:{w.lineno}"
                )
                logger.info(msg)
