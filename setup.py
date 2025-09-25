from setuptools import setup, find_packages

setup(
    name="supra-nexus",
    version="1.0.0",
    author="Supra Foundation LLC",
    description="Advanced reasoning models with transparent thought",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Supra-Nexus/o1",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff"],
        "mlx": ["mlx", "mlx-lm"],
        "gym": ["zoo-gym"],
    },
)
