from setuptools import setup, find_packages

"""Setup for pip package."""

import importlib.util
import os
import setuptools

spec = importlib.util.spec_from_file_location('package_info', 'megatron/core/package_info.py')
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)


__contact_emails__ = package_info.__contact_emails__
__contact_names__ = package_info.__contact_names__
__description__ = package_info.__description__
__download_url__ = package_info.__download_url__
__homepage__ = package_info.__homepage__
__keywords__ = package_info.__keywords__
__license__ = package_info.__license__
__package_name__ = package_info.__package_name__
__repository_url__ = package_info.__repository_url__
__version__ = package_info.__version__


if os.path.exists('megatron/core/README.md'):
    with open("megatron/core/README.md", "r", encoding='utf-8') as fh:
        long_description = fh.read()
    long_description_content_type = "text/markdown"

else:
    long_description = 'See ' + __homepage__
    long_description_content_type = "text/plain"


###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

def req_file(filename, folder="megatron/core"):
    with open(os.path.join(folder, filename), encoding='utf-8') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]

install_requires = req_file("requirements.txt")

###############################################################################

setuptools.setup(
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    # The project's main homepage.
    url=__repository_url__,
    download_url=__download_url__,
    # Author details
    author=__contact_names__,
    author_email=__contact_emails__,
    # maintainer Details
    maintainer=__contact_names__,
    maintainer_email=__contact_emails__,
    # The licence under which the project is released
    license=__license__,
    classifiers=[
        # How mature is this project? Common values are
        #  1 - Planning
        #  2 - Pre-Alpha
        #  3 - Alpha
        #  4 - Beta
        #  5 - Production/Stable
        #  6 - Mature
        #  7 - Inactive
        'Development Status :: 5 - Production/Stable',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',
        # Supported python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(
    include=['megatron.core', 'megatron.core.*'],
    ),
    # Add in any packaged data.
    include_package_data=True,
    # PyPI package information.
    keywords=__keywords__,
)
