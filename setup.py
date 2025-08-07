from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path):
    """
    This function reads a requirements file and returns a list of packages.
    It ignores comments and empty lines.
    """

    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ml_project',
    version='0.0.1',
    author='Manoj Saligrama Harisha',
    author_email="shmanoj2002@gmail.com",
    description='A machine learning project for Black Friday Sales Prediction.',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)