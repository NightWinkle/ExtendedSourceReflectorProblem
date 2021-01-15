import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reflector_problem-NightWinkle", # Replace with your own username
    version="0.0.1",
    author="Guillaume Chazareix",
    author_email="guillaume@chazareix.net",
    description="A package to find solutions of the extended source reflector problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NightWinkle/reflector_problem",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)