from setuptools import setup

setup(
    name='geospatial_fm',
    version='0.1.0',    
    description='MMSegmentation classes for Sen1Floods11',
    author='Paolo Fraccaro, Carlos Gomes, Johannes Jakubik',
    packages=['geospatial_fm'],
    license="Apache 2",
    long_description=open("README.md").read(),
    install_requires=
    [
        "mmsegmentation==0.30",
        "rasterio",
        "einops",
        "timm==0.4.12",
        "tensorboard"
    ]
)