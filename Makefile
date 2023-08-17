PROJECT_NAME = vision_mtl
PYTHON_INTERPRETER = python

requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 $(PROJECT_NAME)

.PHONY: clean data lint requirements
