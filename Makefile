.PHONY: install smoke

install:
	pip install -r requirements.txt

smoke:
	export PYTHONPATH=$$PWD/src && python scripts/smoke_test.py && python scripts/run_experiment.py --manifest data/smoke/manifest.csv --output-dir outputs/smoke
