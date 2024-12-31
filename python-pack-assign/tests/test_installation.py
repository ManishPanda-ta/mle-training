# def test_package_installation():
#     try:
#         from src import python_package
#         assert True
#     except ImportError:
#         assert False


def test_package_installation():
    try:
        import os
        import sys

        sys.path.insert(
            0,
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../src")
            ),
        )
        import python_package

        assert True
    except ImportError:
        assert False
