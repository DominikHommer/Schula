## General

Python Version < 3.13 is needed!

Also git lfs is needed!

Install in extra step (somehow it does not work in requirements file):
- ultralytics==8.3.115
- tensorflow==2.19.0

- Run `streamlit src/main_app.py` to start the Webapp 
- Run `python3 src/main.py` to start a pipeline

## Linting

- Run `pylint src/` to find errors and fix them accordingly.
- Documentation can be found here: https://docs.pylint.org/

## Unit Tests

- Each Module should be covered with a Unit Test
- Pipelines should be tested where appropiate
- To run tests: `cd src` and then run `python -m unittest`
- Tests should be in the folder `tests`
- Mockup test data should be under `tests/fixtures`
- A test is only complete if every aspect of a module / pipeline is tested!

## UI Tests

- TBD