from setuptools import find_packages, setup

setup(
    name="deep_bac",
    packages=find_packages(),
    package_data={
        "gene_set_discovery": ["py.typed", "**/*.json", "**/*.yaml"],
    },
    install_requires=[
        # "dataclasses~=0.6",
        # "dataclasses-json~=0.5.7",
        # "numpy~=1.23.4",
        # "pytorch-lightning~=1.7.7",
        # "ray~=2.1.0",
        # "scanpy~=1.9.1",
        # "scArches~=0.5.4",
        # "scikit-misc~=0.1.4",
        # "scvi-tools~=0.19.0",
        # "torch~=1.13.0",  # nightly version for mac silicon
        # "torchmetrics~=0.10.0",
        # "typed-argument-parser~=1.7.2",
    ],
    extras_require={
        "testing": [
            "coverage~=6.5.0",
            "pytest~=7.1.3",
            "pytest-mock~=3.10.0",
        ]
    },
)