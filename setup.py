from setuptools import setup, find_packages

setup(
    name="",
    version="0.1.0",
    description="Evolutionary Meta-Learning Framework",
    author="The Genie Project",
    author_email="pat@xrai.com",
    url="https://github.com/The-Genie-Project/XRAI",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
        ],
    },
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
) 
