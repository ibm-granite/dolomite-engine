from setuptools import find_packages, setup


try: # for pip >= 10
    from pip._internal.req import parse_requirements
    install_reqs = parse_requirements('requirements.txt', session='packaging')
    reqs = [str(ir.requirement) for ir in install_reqs]
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
    install_reqs = parse_requirements('requirements.txt', session='packaging')
    reqs = [str(ir.req) for ir in install_reqs]


VERSION = "0.0.1.dev"

setup(
    name="dolomite-engine",
    version=VERSION,
    install_requires=reqs,
    author="Mayank Mishra",
    author_email="mayank.mishra2@ibm.com",
    url="https://github.com/ibm-granite/dolomite-engine",
    packages=find_packages("./"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
)
