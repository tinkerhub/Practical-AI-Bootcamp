from setuptools import setup, find_packages

requirements = [
    "flask",
    "requests",
    "tensorflow",
    "tensorflow-datasets",
    "matplotlib",
    "numpy",
    "pytest",
    "gunicorn"
]

setup(
    name="digiter",
    version="0.0.1",
    include_package_data=True,
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
)