setup-venv:
	pip install -e pyartm
	pip install -e pyartm_datasets
	pip install -e pyartm_experiments

install-dev:
	pip install -r dev-requirements.txt
