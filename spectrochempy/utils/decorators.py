from functools import partial


class _dataset_method(object):
    def __init__(self, method, **kwargs):
        self.method = method

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)

    def __call__(self, obj, *args, **kwargs):
        return self.method(obj)


# wrap _set_output to allow for deferred calling
def set_dataset_method(method=None, **kwargs):
    if method:
        # case of the decorator without argument
        return _dataset_method(method)
    else:
        # and with argument
        def wrapper(method):
            return _dataset_method(method, **kwargs)

        return wrapper
