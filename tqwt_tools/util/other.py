import os


def get_package_path() -> str:
    import tqwt_tools

    return os.path.abspath(
        os.path.dirname(tqwt_tools.__file__)
    )  # the abspath of the dirname of the __init__.py file
