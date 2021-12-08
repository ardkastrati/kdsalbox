from setuptools import setup
from importlib.machinery import SourceFileLoader

with open('README.md') as file:
    long_description = file.read()

version = SourceFileLoader('fastsaliency_toolbox.version', 'fastsaliency_toolbox/version.py').load_module()

setup(
   name='fastsaliency_toolbox',
   version=version.version,
   description='A saliency toolbox with efficient saliency models.',
   author='Ard Kastrati',
   author_email='akastrati@adobe.com',
   url='https://git.corp.adobe.com/adobe-research/fastsaliency_toolbox/fastsaliency_toolbox',
   packages=['fastsaliency_toolbox'],
   long_description=long_description,
   long_description_content_type='text/markdown',
   keywords='saliency, fast saliency, efficient models, matlab, knowledge distillation, toolbox',
   license='Copyright Adobe Inc.',
   install_requires=[],
)
