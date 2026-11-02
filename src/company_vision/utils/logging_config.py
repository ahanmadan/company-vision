
import logging
import sys

from company_vision.core.settings import Settings


def configure_logging(settings: Settings) -> None:
    level = getattr(
        logging,
        settings.app.log_level,
    )

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
