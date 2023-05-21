from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
VERSION = '0.0.2'
NAME = 'vocabtrimmer'
LICENSE = 'MIT License'
setup(
    name=NAME,
    packages=find_packages(exclude=['assets','tests', 'misc', 'asset', "experiments", "hf_operations", "metric_files"]),
    version=VERSION,
    license=LICENSE,
    description='Trimming vocabulary of pre-trained multilingual language models to language localization.',
    url='https://github.com/asahi417/lm-vocab-trmmer',
    download_url="https://github.com/asahi417/lm-vocab-trmmer/archive/v{}.tar.gz".format(VERSION),
    keywords=['language model', 't5', 'gpt3', 'bert' 'nlp', 'multilingual', 'efficient-model'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        f'License :: OSI Approved :: {LICENSE}',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "torch",
        "tqdm",
        "requests",
        "transformers",
        "sentencepiece",
        "tokenizers"
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'vocabtrimmer-trimming = vocabtrimmer.cl.trimming:main'
        ]
    }
)

