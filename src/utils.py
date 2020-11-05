from . import simple_logging as logging
import torch

log = logging.getLogger()


def init():
    log.info('Initializing')
    torch.cuda.device(0)
    log.info(f'Using CUDA device {torch.cuda.get_device_name(torch.cuda.current_device())}')
