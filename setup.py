from setuptools import setup, find_packages

setup(
    name="thinkbrake",
    version="0.1",
    packages=find_packages(include=["thinkbrake", "thinkbrake.*"]),
    install_requires=[
        "sglang>=0.3.0",
        "tqdm",
    ],
    extras_require={
        "bfcl": [
            "sglang>=0.3.0",
            "tqdm",
        ],
    },
)
