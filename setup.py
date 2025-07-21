from setuptools import setup, find_packages

setup(
    name="supply-chain-anomaly-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "PyYAML>=6.0",
        "requests>=2.27.0",
        "imbalanced-learn>=0.8.0",
        "mlflow>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "databricks": [
            "mlflow>=2.0.0",
            "delta-spark>=2.0.0",
            "pyspark>=3.3.0",
        ],
    },
    author="JEDDI Mohammed",
    author_email="mohammed.jeddi@hp.com",
    description="A machine learning system for detecting and classifying supply chain anomalies",
    keywords="supply-chain, anomaly-detection, machine-learning, mlflow",
    url="https://github.azc.ext.hp.com/mohammed-jeddi/supply-chain-anomaly-detection",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)