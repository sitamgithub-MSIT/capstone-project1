# Importing required libraries
from setuptools import setup, find_packages

# Defining constants
HYPHEN_E_DOT = "-e ."


def get_requirements(filename: str) -> list[str]:
    """Get requirements from requirements.txt file"""

    requirements = []

    with open(filename) as f:
        requirements = f.readlines()
        requirements = [x.replace("\n", "") for x in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# Defining setup
setup(
    name="midterm-project",
    version="0.0.1",
    author="Sitam Meur",
    author_email="sitammeur@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    description="Midterm Project for ML Zoomcamp 2023 by Alexey Grigorev",
)
