import os

from plangolin.logging_setup import get_logger

logger = get_logger(__name__)


def exit_if_orphaned():
    # concurrent.futures.ProcessPoolExecutors are suffering from non-exiting workers once the parent process is gone
    # see issue https://github.com/python/cpython/issues/111873
    import multiprocessing

    # wait for parent process to die first; may never happen
    multiprocessing.parent_process().join()
    os._exit(-1)


def increase_resource_limit(fraction: float = 1.0):
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != hard:
        new_soft = min(abs(int(fraction * hard)), hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logger.info(
            f"Changing resource limits to: [{soft = } --> {new_soft = }, {hard = }]"
        )
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.info(f"Resource limits: [{soft = }, {hard = }]")
