from setuptools import find_packages, setup

setup(
    name="ltsf",
    version="0.1",
    description="Long-term time-series forecasting package",
    url = "https://github.com/Hank0626/ltsf",
    author="Peiyuan Liu",
    author_email="hank010626@163.com",
    packages=find_packages(),
    python_requires=">=3.7.0",
    install_requires=["torch==1.12.0+cu116", "reformer-pytorch", "tqdm", "pyyaml", "einops==0.4.0",
"numpy==1.23.5", "pandas==1.5.3", "scikit-learn==1.2.2", "scipy==1.10.1", "sympy==1.11.1", "patool==1.12", "sktime==0.16.1"],
)
