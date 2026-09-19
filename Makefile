.PHONY: train test

train:
	python src/models/train.py

test:
	pytest tests/
