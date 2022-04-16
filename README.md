# Lesson Overview

In this lesson, we will apply the skills acquired in the [Machine Learning Fundamentals and Decision Trees](https://github.com/ivanovitchm/ppgeecmachinelearning/tree/main/lessons/week_02/sources) lesson to deploy a classification model on publicly available [Census Bureau data](http://archive.ics.uci.edu/ml/datasets/Adult). 

We will deploy the model using the [FastAPI](https://fastapi.tiangolo.com/) package and create API tests. The API tests will be incorporated into a CI/CD framework using GitHub Actions. After we build our API locally and test it, we will deploy it to [Heroku](https://www.heroku.com/) and test it again once live. [Weights & Biases](https://wandb.ai/) will be used to manage and tracking all artifacts.

## :arrow_forward: Environment Setup

Create a conda environment with ``environment.yml``:

```bash
conda env create --file environment.yml
```

To remove an environment in your terminal window run:

```bash
conda remove --name myenv --all
```

To list all available environments run:

```bash
conda env list
```

To activate the environment, use

```bash
conda activate myenv
```