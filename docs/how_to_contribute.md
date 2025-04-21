# How to Contribute to Documentation?

## Basics

Our static html pages are built using `sphinx`. The pages should be written in `.rst` format. The language should be clear, simple and precise.

## Contents

```
build - here are stored html pages (is ignored by git)
source - here are stored rst files
```

## How to Write Docstrings

There are 2 styles of docstrings that are supported by Sphinx:

* Google style (preferable as most our scripts are already written according this style, [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html))
```
Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value
```

* Numpy style

```
Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value
```

In docstrings the functions, methods, etc. can be referenced. See the Chatsky example:

```
It passes :py:meth:`_run_pipeline` to :py:attr:`messenger_interface` as a callback,
so every time user request is received, :py:meth:`_run_pipeline` will be called.
```

**Note:** the docstring style should also be present in naming. There are few tips to be consistent in naming:

1. Use verb infitive to name function or method (use `add` instead of `adds`)

2. Use verb finite or name phrase to name classes (use `adds` or `(base) class for`)

## Sphinx

The pages are build running

```bash
python -m poetry run sphinx-build -b html docs/source/ docs/build/
```

**Note**: sphinx is included in the poetry enviroment. If you haven't yet installed poetry dependences, install them first.

**Links**:

[1] - [Sphinx documentation](https://www.sphinx-doc.org/en/master/tutorial/index.html)

[2] - [Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)

## ReStructuredText (.rst)

ReStructuredText is a file format for textual data used for technical documentation. To create or update `rst` files go to the `docs/source` directory.

**Links**:

[1] - [Practical guide](https://www.writethedocs.org/guide/writing/reStructuredText/)

[2] - [Cheat sheet](https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst#cit2002)