# setup.py
from setuptools import setup, find_packages

setup(
    name="tabul",
    version="0.0.1",
    description="A Python package for unlearning Tabular Generative Model",
    author="Jeroen Galjaard",
    author_email="J.M.Galjaard@tudelft.nl",
    url="https://gitlab.ewi.tudelft.nl/dmls/publications/tabul-rasa",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "accelerate",
        "prdc==0.2",
        "datasets",
        "transformers",
        "sdmetrics==0.19.0",
        "numpy==1.25.0",
        "pandas==2.2.3",
        "xlrd>=2.0.1",
        "category_encoders~=2.8.1",
        "prdc~=0.2",
        "xgboost~=3.0.0",
        "hydra-core~=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<3.13",
)
