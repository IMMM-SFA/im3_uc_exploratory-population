from setuptools import setup, find_packages


def get_requirements():
    """Get a list of required packages to install.  Includes GitHub packages as well."""

    installs = []
    depends = []

    with open('requirements.txt') as f:

        for i in f.readlines():

            pkg = i.strip().split('|')
            pkg_len = len(pkg)

            if pkg_len == 1:
                installs.append(pkg[0])

            elif pkg_len == 2:
                installs.append(pkg[0])
                depends.append(pkg[1])

            else:
                raise IndexError("Too many arguments entered in the 'requirements.txt' file for a single line:  {}".format(i))

        return installs, depends


# read in requirements
install_list, depends_list = get_requirements()


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
    install_requires=install_list,
    dependency_links=depends_list,
    include_package_data=True
)
