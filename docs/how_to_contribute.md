# How to Contribute to Documentation?

## Basics

Our static html pages are built using `sphinx`. The pages should be written in `.rst` format. The language should be clear, simple and precise.

## Contents

```
build - here are stored html pages
source - here are stored rst files
```

## Sphinx

The pages are build running

```bash
python -m poetry run sphinx-build -M html docs/source/ docs/build/
```

**Note**: sphinx is included in the poetry enviroment. If you are not yet installed poetry dependences, install them first.

**Links**:

[1] - [Sphinx documentation](https://www.sphinx-doc.org/en/master/tutorial/index.html)
[2] - [Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)

## ReStructuredText (.rst)

ReStructuredText is a file format for textual data used for technical documentation. To create or update `rst` files go to the `docs/source` directory.

**Links**:

[1] - [Practical guide](https://www.writethedocs.org/guide/writing/reStructuredText/)
[2] - [Cheat sheet](https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst#cit2002)