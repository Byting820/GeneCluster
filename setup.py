from setuptools import setup, find_packages

setup(
    name="GeneCluster",
    version="0.1.1",
    author="Byting",
    author_email="yutingya820@163.com",
    description="A gene clustering model using deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Byting820/GeneCluster",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "numba",
        "pandas",
        "scanpy",
        "h5py",
        "sklearn",
        "umap-learn",
        "matplotlib",
        "faiss"
    ],
    entry_points={
        "console_scripts": [
            "GeneCluster=GeneCluster.main:run",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)