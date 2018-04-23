from setuptools import setup

setup(
    name='chainerltr',
    version='0.1.0',
    description='Neural Learning to Rank using Chainer',
    url='https://github.com/rjagerman/chainerltr',
    download_url = 'https://github.com/rjagerman/chainerltr/archive/v0.1.0.tar.gz',
    author='Rolf Jagerman',
    author_email='rjagerman@gmail.com',
    license='MIT',
    packages=['chainerltr',
              'chainerltr.clickmodels',
              'chainerltr.evaluation',
              'chainerltr.functions',
              'chainerltr.loss',
              'test',
              'test.clickmodels',
              'test.evaluation',
              'test.examples',
              'test.loss'],
    install_requires=['numpy>=1.13.0',
                      'chainer>=2.0.0',
                      'scikit-learn>=0.19.1',
                      'scipy>=1.0.0'],
    test_suite='nose.collector',
    tests_require=['nose']
)

