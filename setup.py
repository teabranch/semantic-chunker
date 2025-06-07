from setuptools import setup, find_packages

setup(
    name='simple-semantic-chunker',
    version='0.1.0',
    author='TeaBranch',
    author_email='your.email@example.com', # Replace with a valid email
    description='A simple library to split documents into semantically coherent chunks using OpenAI embeddings.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TeaBranch/simple-semantic-chunker', # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        'numpy',
        'openai>=1.0.0' # Specify a version that supports the new client
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License', # Assuming MIT, update if different
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    keywords='semantic chunking text processing nlp openai embeddings',
)
