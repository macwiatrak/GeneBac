from setuptools import find_packages, setup

setup(
    name="deep_bac",
    packages=find_packages(),
    package_data={
        "gene_set_discovery": ["py.typed", "**/*.json", "**/*.yaml"],
    },
    install_requires=[
        "dataclasses~=0.6",
        "dataclasses-json~=0.5.7",
        "einops~=0.6.0",
        "fastaparser~=1.1.1",
        "gffpandas~=1.2.0",
        "huggingface-hub~=0.11.1",
        "ipython~=8.7.0",
        "jupyterlab==3.5.1",
        "matplotlib==3.6.2",
        "numpy~=1.23.5",
        "pyarrow==10.0.1",
        "pyfastx~=0.8.4",
        "pytorch-lightning~=1.8.6",
        "tensorboardX~=2.5.1",
        "torch~=1.13.1",
        "torchmetrics~=0.11.0",
        "tqdm~=4.64.1",
        "transformers~=4.25.1",
        "typed-argument-parser~=1.7.2",
    ],
    extras_require={
        "testing": [
            "coverage~=6.5.0",
            "pytest~=7.1.3",
            "pytest-mock~=3.10.0",
        ]
    },
)