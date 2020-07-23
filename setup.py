import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sarkas-murillogroup", # Replace with your own username
    version="0.0.1",
    author="Murillo Group MSU",
    author_email="sarkasdev@gmail.com",
    description="A Fast Pure-Python Molecular Dynamics Toolkit for Non-Ideal Plasmas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://murillo-group.github.io/sarkas",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'numba>=0.50.1',
        'fdint',
        'tqdm',
        'pyfiglet',
        'pyyaml'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
