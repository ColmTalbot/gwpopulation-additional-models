# gwpopulation-additional-models
Repository for dumping additional models for use with gwpopulation that I don't want to add to the main package.

This is also intended as a template for downstream users to implement models and provide a minimal packaging example.

## How do I use this as a template?

- Create a copy of this repository using the "Use this template" button on GitHub.
- Go through `setup.cfg`, `pyproject.toml` and update to match your package name.
- Replace the `gwpopulation_additional_models` directory with one matching your package name. Make sure to include the `__init__.py` file for packaging.
- Add your model to this new directory.
- If you use `gwpopulation.utils.xp` or `gwpopulation.utils.scs` in your model, you will need to update the `[options.entry_points]` in `setup.cfg`.
- Update `requirements.txt` to add any additional packages you need.
- Write meaningful unit tests in the `tests` directory, these will run automatically via GitHub actions.
- If you want to release the package to `pypi`, you need to set your access token using [a secret](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions). Versions can then be pushed to pypi by creating a tag and release. The version of the code is automatically set using `setuptools_scm`.

## Why should I use this template?

This template is set up to automate packaging and accessing the `GWPopulation` backend.
This is intended for users who are looking to share their models as published packages, which is [always a good idea](https://pythonpackaging.info/01-Introduction.html).