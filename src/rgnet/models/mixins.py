import torch


class DeviceAwareMixin:
    """
    Mixin to register the device of the module.
    This is useful for modules that need to know the device they are on.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("_device_register", torch.empty(1))

    @property
    def device(self) -> torch.device:
        return self._device_register.device
