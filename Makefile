.PHONY: all train report test replay api serve test-serving
all: test-serving
train:
	@echo 'Run train_from_mongo.bat when the new MongoDB collection is ready.'
report:
	@echo 'Current results: reports/latest_training.md'
test:
	python -m pytest -q tests
replay:
	@echo 'python -m ship_fault.cli replay-csv "/path/to/recording.csv" --batch-size 2000'
api:
	python -m ship_fault.cli serve --host 127.0.0.1 --port 8000
serve:
	python -m ship_fault.cli serve --host 127.0.0.1 --port 8000
test-serving:
	python -m pytest -q -p no:cacheprovider tests/test_serving.py
