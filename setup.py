from setuptools import setup

setup(
    name='semlm',
    version='0.1',
    author='Ben Lambert',
    author_email='ben@benjaminlambert.com',
    packages=['semlm'],
    description='Semantic language modeling',
    keywords=['semantic language modeling', 'language model', 'asr'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License"],
    install_requires=['asr_tools', 'sklearn', 'numpy', 'scipy'],
    test_suite='test.test_main.Testing'
)
