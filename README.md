# chipseq-utils

# Create the python package

Take a look at setup.cfg and setup.py.

The project can be installed from this directory with

```python
pip install .
```

and to include development and documentation dependencies, please use

```python
pip install .[dev,doc]
```

# Create documentation

## Sphinx quickstart

Sphinx is a package used to make documentation for python projects.

After installing the project with the `doc` extra, the command `sphinx-quickstart` is available. Use the command below to create sphinx documentation in the directory `./docs`. Most python projects put their documentation there.

Select the default option if you are not sure what to enter.

```
sphinx-quickstart docs
```

To generate an HTML version of the documentation, run

```
make html
```

within the `docs` directory.

Then run an HTTP server in `_build/html`

```
cd _build/html
python -m http.server
```

and go to http://0.0.0.0:8000.

## Edit docs/conf.py

The file `docs/conf.py` configures the documentation. That's where you set things like the metadata (eg author, project name), the HTML theme, and any extensions we want to use.

All of the documentation is written in Restructured Text (`.rst`) files or Jupyter Notebooks.

## Add documentation

The landing page of the documentation is written in `index.rst`. From here, you can point to other points in the documentation. The toctree is the table of contents.

One nice layout is having a User Guide and an API Reference section. The API reference is generated automatically (read automagically) from the docstrings in the Python code. The User Guide is written up in rst files or Jupyter Notebooks.

# Code style

For docstrings, I would suggest [the Numpy standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).

For Python code more generally, I would suggest using [the `black` formatter](https://github.com/psf/black). It is written by one of the core Python developers, if that helps.
