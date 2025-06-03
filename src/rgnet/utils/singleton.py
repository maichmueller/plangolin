from typing import Any, Dict


class PickleSafeSingleton(type):
    _instances: Dict[type, Any] = {}

    def __new__(metacls, name, bases, namespace):
        # Inject a __reduce__ into the class so pickle always returns the singleton
        def __reduce__(self):
            # On unpickle, calling the class returns the existing instance
            return (self.__class__, ())

        # Only add if not already defined
        namespace.setdefault("__reduce__", __reduce__)
        return super().__new__(metacls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            inst = super().__call__(*args, **kwargs)
            cls._instances[cls] = inst
        return cls._instances[cls]


# Test pickling
if __name__ == "__main__":
    import pickle

    # Example usage
    class Configuration(metaclass=PickleSafeSingleton):
        def __init__(self, path):
            print("init", path)
            self.path = path

    cfg1 = Configuration("/etc/config")
    cfg2 = Configuration("/other")
    assert cfg1 is cfg2

    data = pickle.dumps(cfg1)
    cfg3 = pickle.loads(data)
    assert cfg3 is cfg1  # Same singleton instance after unpickle
    print("Singleton preserved across pickle in this process")
