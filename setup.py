# from distutils.core import setup
from setuptools import setup, find_packages

# Define requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

description = """
Repositorio donde se realizarán las entregas de los Trabajos Prácticos 1 y 2 para la materia Técnicas 
y Algoritmos de Aprendizaje Automático - Máster en Management & Analytics - ITBA
"""

# Define setup
setup(
    name="src",
    description=description,
    author="Simón P. García Morillo",
    author_email="simongmorillo1@gmail.com",
    version="1.0.0",
    install_requires=requirements,
    packages=find_packages(),
    # package_data={"test": ["test/*"]},
    long_description=open("README.md").read(),
    license=open("LICENSE").read()
)