from setuptools import setup, find_packages

#print(find_packages())

setup(
    name='semlm',
    version='0.1',
    author='Ben Lambert',
    author_email='belambert@mac.com',
    packages=['semlm'],
    # packages = find_packages(),
    description='Semantic language modeling',
    long_description=open('README.md').read(),
    test_suite='nose.collector',
    tests_require=['nose'],
    # test_suites=
    # install_requires=[]
)
