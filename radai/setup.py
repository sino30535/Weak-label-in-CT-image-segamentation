from setuptools import setup

setup(name='radai',
      version='0.1',
      description='image segmentation for CT scans',
      url='https://github.com/sino30535/Rad_AI_technical_test/tree/master/radai',
      author='Shlee',
      author_email='sino30535@gmail.com',
      license='MIT',
      packages=['radai'],
      install_requires=[
          'tensorflow', 'tqdm', 'Keras', 'numpy', 'opencv_python', 'scikit_learn'
      ]
      zip_safe=False)
