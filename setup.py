"""Package the flat project root without imposing a src/ directory."""
from setuptools import find_namespace_packages, setup

setup(
    packages=["dualpd", "dualpd.runtime"] + [
        "dualpd." + name for name in find_namespace_packages(
            ".", include=["data", "data.*"]
        )
    ],
    package_dir={"dualpd": "."},
    exclude_package_data={"": ["**/__pycache__/*", "**/*.pyc"]},
)
