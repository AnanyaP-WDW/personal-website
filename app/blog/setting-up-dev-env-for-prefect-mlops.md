---
title: Setting up MLOps dev environment with Dagster
date: 10-07-2025
description: 
tags: ["MLOps", "Dagster", "data engineering", "scaling"]
---

# Setting up MLOps dev environment with Dagster

## Table of Contents

1. [`Why specialised tooling for MLOps`](#why-specialised-tooling-for-mlops)
2. [`Dagster`](#dagster)
3. [`Dockerfile for workflows`](#dockerfile-for-workflows)
4. [`Repo.py test workflow`](#repo-py-test-workflow)

---

## Why specialised tooling for MLOps

MLOps is a specialised subset of DevOps mainly catering to building pipelines for creating training data, feature engineering, training models, sclaed hyper-parameter tuning, evaluations and benchmarks. The main requirements can be summarised as foolws:
1) Scheduler 
2) Persistent storage and viewing runs in a easy to decipher UI
3) Ability to add triggers, cron etc
4) Effecient hardware utilization

Creting something from scratch is an overhead on the mlops and engineering teams. Thats why teams try to use pre-exisitng tooling that has been tested through oss development. There are many options available in the market like airflow, prefect, dagster, kubeflow, flyte etc. In this blog I will be creaging a dev environment using Dagster.

## Dagster

We will need a relational database like postgres, a object storage service like minio (I/O manager for storing intermediate steps), a daemon for scheduling Dagster workflows and a Dagster webserver. Based on your teams fimiliarity we can keep all the workflow in a single image (not recommende for largeer workflows) or separately. I like to keep each workflow as a separate image, and these in turn can be added to the docker compose. 

```yml
services:
  # This service runs the postgres DB used by dagster for run storage, schedule storage,
  # and event log storage.
  postgresql:
    image: postgres:11
    container_name: postgresql
    environment:
      POSTGRES_USER: 'postgres_user'
      POSTGRES_PASSWORD: 'postgres_password'
      POSTGRES_DB: 'postgres_db'
    networks:
      - dagster_network
    volumes:
      - postgres_data:/var/lib/postgresql/data # volume mapping for postgres data
    healthcheck:
      test: ['CMD-SHELL', 'pg_isready -U postgres_user -d postgres_db']
      interval: 10s
      timeout: 8s
      retries: 5

  # MinIO service for S3-compatible object storage (IO Manager)
  minio:
    image: minio/minio:latest
    container_name: minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: 'minioadmin'
      MINIO_ROOT_PASSWORD: 'minioadmin'
    ports:
      - '9000:9000'  # MinIO API
      - '9001:9001'  # MinIO Console UI
    volumes:
      - minio_data:/data
    networks:
      - dagster_network
    healthcheck:
      test: ['CMD', 'curl', '-f', 'http://localhost:9000/minio/health/live']
      interval: 10s
      timeout: 5s
      retries: 5

  # MinIO client to create bucket on startup
  minio_create_bucket:
    image: minio/mc:latest
    container_name: minio_create_bucket
    depends_on:
      minio:
        condition: service_healthy
    networks:
      - dagster_network
    entrypoint: >
      /bin/sh -c "
      mc alias set myminio http://minio:9000 minioadmin minioadmin;
      mc mb myminio/dagster-storage --ignore-existing;
      mc anonymous set public myminio/dagster-storage;
      echo 'Bucket dagster-storage created successfully';
      "


  # This service runs dagster-webserver, which loads your user code from the user code container.
  # Since our instance uses the QueuedRunCoordinator, any runs submitted from the webserver will be put on
  # a queue and later dequeued and launched by dagster-daemon.
  webserver:
    build:
      context: .
      dockerfile: ./Dockerfile_dagster
    entrypoint:
      - dagster-webserver
      - -h
      - '0.0.0.0'
      - -p
      - '3000'
      - -w
      - workspace.yaml
    container_name: webserver
    expose:
      - '3000'
    ports:
      - '3000:3000'
    env_file:
      - .env
    environment:
      DAGSTER_POSTGRES_USER: 'postgres_user'
      DAGSTER_POSTGRES_PASSWORD: 'postgres_password'
      DAGSTER_POSTGRES_DB: 'postgres_db'
      # MinIO Configuration for IO Manager
      MINIO_ENDPOINT: 'http://minio:9000'
      MINIO_ACCESS_KEY: 'minioadmin'
      MINIO_SECRET_KEY: 'minioadmin'
      MINIO_BUCKET: 'dagster-storage'
    volumes: # Make docker client accessible so we can terminate containers from the webserver
      - /var/run/docker.sock:/var/run/docker.sock
      - /tmp/io_manager_storage:/tmp/io_manager_storage
      - ./workspace.yaml:/opt/dagster/app/workspace.yaml:ro
      # Mount volume for S3 downloads
      - ./s3_downloads:/opt/dagster/s3_downloads
    networks:
      - dagster_network
    depends_on:
      postgresql:
        condition: service_healthy
      minio:
        condition: service_healthy
      test_workflow:
        condition: service_started

  # This service runs the dagster-daemon process, which is responsible for taking runs off of the queue and launching them, as well as creating runs from schedules or sensors.
  daemon:
    build:
      context: .
      dockerfile: ./Dockerfile_dagster
    entrypoint:
      - dagster-daemon
      - run
    container_name: daemon
    restart: on-failure
    env_file:
      - .env
    environment:
      DAGSTER_POSTGRES_USER: 'postgres_user'
      DAGSTER_POSTGRES_PASSWORD: 'postgres_password'
      DAGSTER_POSTGRES_DB: 'postgres_db'
      # MinIO Configuration for IO Manager
      MINIO_ENDPOINT: 'http://minio:9000'
      MINIO_ACCESS_KEY: 'minioadmin'
      MINIO_SECRET_KEY: 'minioadmin'
      MINIO_BUCKET: 'dagster-storage'
    volumes: # Make docker client accessible so we can launch containers using host docker
      - /var/run/docker.sock:/var/run/docker.sock
      - /tmp/io_manager_storage:/tmp/io_manager_storage
      - ./workspace.yaml:/opt/dagster/app/workspace.yaml:ro
      # Mount volume for S3 downloads
      - ./s3_downloads:/opt/dagster/s3_downloads
    networks:
      - dagster_network
    depends_on:
      postgresql:
        condition: service_healthy
      minio:
        condition: service_healthy
      test_workflow:
        condition: service_started

  test_workflow:
    build:
      context: .
      dockerfile: ./Dockerfile_workflow
    container_name: test_workflow
    image: test_workflow_image
    restart: always
    environment:
      DAGSTER_POSTGRES_USER: "postgres_user"
      DAGSTER_POSTGRES_PASSWORD: "postgres_password"
      DAGSTER_POSTGRES_DB: "postgres_db"
      DAGSTER_CURRENT_IMAGE: "test_workflow_image"
    networks:
      - dagster_network

networks:
  dagster_network:
    driver: bridge
    name: dagster_network

volumes:
  minio_data:
    driver: local
  postgres_data:
    driver: local

```

## Dockerfile for workflows

Here is an example Dockerfile for the `test_workflow` service. This image contains your Dagster user code.

```Dockerfile
FROM python:3.10-slim

# Checkout and install dagster libraries needed to run the gRPC server
# We need to pin the version of dagster-graphql and dagster-webserver
# to the same version as dagster
RUN pip install \
    dagster \
    dagster-postgres \
    dagster-docker \
    dagster-k8s \
    dagster-aws \
    dagster-celery[flower,redis,k8s]

# Add repository code
WORKDIR /opt/dagster/app
COPY . /opt/dagster/app

# Run dagster gRPC server on port 4000
EXPOSE 4000

# CMD allows this to be overridden from run launchers or executors that want
# to run other commands against your code
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-f", "repo.py"]
```

## Repo.py test workflow

Here is an example `repo.py` file that defines an example job.

```python
from dagster import job, op, repository

@op
def hello():
    return "Hello, Dagster!"

@job
def hello_job():
    hello()

@repository
def my_repository():
    return [hello_job]
```

