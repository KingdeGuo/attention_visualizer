from setuptools import setup, find_packages

setup(
    name="attention_visualizer",
    version="0.1.0",
    description="Transformer Attention Mechanism Visualizer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "plotly>=5.0.0", 
        "dash>=2.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
