from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="lie_discriminator",
    version="0.1",
    description="A machine learning project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rail",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.10,<3.11",
)
