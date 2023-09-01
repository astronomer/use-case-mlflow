Predict possum tail length using linear regression with MLflow and Airflow
==========================================================================

This repository contains the DAG code used in the [Regression with Airflow + MLflow use case example](https://docs.astronomer.io/learn/use-case-airflow-mlflow). 

The DAGs in this repository use the following packages:

- [MLflow Airflow provider](https://github.com/astronomer/airflow-provider-mlflow)
- [MLflow Python package](https://pypi.org/project/mlflow/)
- [Amazon Airflow provider](https://registry.astronomer.io/providers/apache-airflow-providers-amazon/versions/latest)
- [Astro Python SDK](https://registry.astronomer.io/providers/astro-sdk-python/versions/latest)

# How to use this repository

This section explains how to run this repository with Airflow. Note that you will need to copy the contents of the `.env_example` file to a newly created `.env` file. No external connections are necessary to run this repository locally, but you can add your own credentials in the file if you wish to connect to your tools. 

## Option 1: Use GitHub Codespaces

Run this Airflow project without installing anything locally.

1. Fork this repository.
2. Create a new GitHub codespaces project on your fork. Make sure it uses at least 4 cores!
3. After creating the codespaces project the Astro CLI will automatically start up all necessary Airflow components and the local MinIO and MLflow instances. This can take a few minutes. 
4. Once the Airflow project has started, access the Airflow UI by clicking on the **Ports** tab and opening the forward URL for port 8080. The MLflow instance is accessible at port 5000, the MinIO instance at port 9000.

## Option 2: Use the Astro CLI

Download the [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) to run Airflow locally in Docker. `astro` is the only package you will need to install locally.

1. Run `git clone https://github.com/astronomer/use-case-mlflow.git` on your computer to create a local clone of this repository.
2. Install the Astro CLI by following the steps in the [Astro CLI documentation](https://docs.astronomer.io/astro/cli/install-cli). Docker Desktop/Docker Engine is a prerequisite, but you don't need in-depth Docker knowledge to run Airflow with the Astro CLI.
3. Run `astro dev start` in your cloned repository.
4. After your Astro project has started. View the Airflow UI at `localhost:8080`, the MLflow UI at `localhost:5000` and the MinIO UI at `localhost:9000`.

## Resources

- [Predict possum tail length using linear regression with MLflow and Airflow use case](https://docs.astronomer.io/learn/use-case-airflow-mlflow).
- [Use MLflow with Apache Airflow tutorial](https://docs.astronomer.io/learn/airflow-mlflow).
- [MLflow documentation](https://mlflow.org/docs/latest/index.html).
- [MLflow Airflow provider repository](https://github.com/astronomer/airflow-provider-mlflow).
- [Astro Python SDK tutorial](https://docs.astronomer.io/learn/astro-python-sdk).
- [Astro Python SDK documentation](https://astro-sdk-python.readthedocs.io/en/stable/index.html).
- [Astro Python SDK repository](https://github.com/astronomer/astro-sdk).