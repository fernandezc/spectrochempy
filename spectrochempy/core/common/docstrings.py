import textwrap
import docrep

__all__ = ["DocstringProcessor"]


class DocstringProcessor(docrep.DocstringProcessor):
    def __init__(self, **kwargs):
        object = kwargs.pop("object", "object")
        super().__init__(**kwargs)
        params = {
            "inplace": "inplace : bool, optional, default: False"
            f"\n    By default, the method returns a newly allocated {object}."
            f"\n    If `inplace` is set to True, the input {object} is returned.",
            "kwargs": "**kwargs"
            "\n    Optional keyword parameters. See Other Parameters.",
            "out": f"{object}"
            f"\n    Input {object} or a newly allocated {object}"
            "\n    depending on the `inplace` flag.",
            "new": f"{object}\n    Newly allocated {object}.",
        }
        self.params.update(params)

    def dedent(self, s, stacklevel=3):
        s_ = s
        start = ""
        end = ""
        string = True
        if not isinstance(s, str) and hasattr(s, "__doc__"):
            string = False
            s_ = s.__doc__
        if s_.startswith("\n"):  # restore the first blank line
            start = "\n"
        if s_.strip(" ").endswith("\n"):  # restore the last return before quote
            end = "\n"
        s_mod = super().dedent(s, stacklevel=stacklevel)
        if string:
            s_mod = f"{start}{s_mod}{end}"
        else:
            s_mod.__doc__ = f"{start}{s_mod.__doc__}{end}"
        return s_mod


def add_docstring(*args):
    """
    Decorator which add a docstring to the actual func doctring.
    """

    def new_doc(func):

        for item in args:
            item.strip()

        func.__doc__ = textwrap.dedent(func.__doc__).format(*args)
        return func

    return new_doc
