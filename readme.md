# CAS Practical Machine Learning - Project Thesis

Project thesis for [CAS - Practical Machine Learning](https://www.bfh.ch/de/weiterbildung/cas/practical-machine-learning/). 

## Resources

## Dev Setup / How-Tos

### Setup after cloning the repository

1. Sync Python project dependencies using uv:

    ```
    uv sync
    ```

1. Rebuild the Jupyter notebooks by running the following command:

    ```
    jupytext --sync .\notebooks\*.py
    ```

### Jupyter Notebooks

- [Using uv with Jupyter from VSCode](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code)
- [Jupytext](https://jupytext.org/)
  - [Jupytext Sync Extension for VSCode](https://jupytext.org/integrations/vs-code/)
