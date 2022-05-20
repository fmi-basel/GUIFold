import setuptools

setuptools.setup(
     name='guifold',
     version='0.1',
     author="Georg Kempf",
     zip_safe=False,
     packages=setuptools.find_packages(),
     include_package_data=True,
     scripts=['guifold/afgui.py', 'guifold/afeval.py'],)
