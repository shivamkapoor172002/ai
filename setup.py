from setuptools import setup, find_packages

setup(
    name='streamlit_app',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ],
)
