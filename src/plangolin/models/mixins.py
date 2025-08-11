import torch


class DeviceAwareMixin:
    """
    Mixin to register the device of the module.
    This is useful for modules that need to know the device they are on.

    All arguments are simply forwarded to the parent class's `__init__` method.

    Note:
        This mixin has to be placed before the `torch.nn.Module` (or Module deriving classes)
        in the inheritance chain. For 2 reasons:
            - torch.nn.Module's `__init__` method will need to hvae been called before registering the buffer
            (ensured by this mixin)
            - torch.nn.Module does not call a super().__init__() in its `__init__` method
            (can be enforced via argument, but creates problems of their own)
        For this, consider also the super() calls made by the parent classes.
        To be safe, always place this mixin as the first parent class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_device_register", torch.empty(1), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_register.device
