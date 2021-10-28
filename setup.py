from setuptools import setup

setup(
    name='planner',
    packages=['planner_service'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)

