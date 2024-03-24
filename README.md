1. Install the 'dev' dependencies for the 'aera02-aisia' package: `pip install .[dev]` or `pre-commit install` these two packages: `pre-commit` and `ruff`.
2. In the `aera02-aisia` root directory (where `.pre-commit-config.yaml` is), run the pre-commit installation: `pre-commit install`
    1. This will update your `./.git/hooks/pre-commit` which will automatically run commands on any staged files when you run git commit in this project
    2. The pre-commit hooks I've configured are: `ruff` and `ruff-format`
        1. `ruff`: Lints (checks syntax) your python code for potential errors or non-compliance with most of PEP8 standards
        2. `ruff-format`: Automatically reformats your python code to comply with standards. Note that this will change your files, so you'll have to re-add them to the staging area with `git add`
3. You can manually run `pre-commit` or `ruff` like so:
```
    pre-commit run --files <FILEPATHS>
    ruff <FILEPATHS>
```

