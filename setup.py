from setuptools import setup, find_packages


setup(
    name='sa_popgrid',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/sa_popgrid.git',
    license='BSD 2-Clause',
    author='Chris R. Vernon',
    author_email='chrisrvernon@gmail.com',
    description='Code and process for conducting sensitivity analysis for the gridded population gravity model on a cluster',
    python_requires='>=3.6',
    install_requires=[
        'rasterio~=1.1.5',
        'simplejson~=3.17.0',
        'numpy~=1.19.5',
        'pandas~=1.2.4',
        'xarray~=0.18.2',
    ],
    include_package_data=True
)
