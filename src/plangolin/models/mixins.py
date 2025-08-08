import torch


class DeviceAwareMixin:
    """
    Mixin to register the device of the module.
    This is useful for modules that need to know the device they are on.

    Note:
        This mixin has to be placed before the `torch.nn.Module` in the inheritance chain.
        For this, consider also the super() calls made by the parent classes.
        To be safe, always place this mixin as the first parent class.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("_device_register", torch.empty(1))

    @property
    def device(self) -> torch.device:
        return self._device_register.device
