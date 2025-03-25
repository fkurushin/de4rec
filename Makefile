VENV=.venv
SOURCES=src
TESTS=tests


# Installation

.venv:
	/usr/bin/python3 -m venv .venv
	. .venv/bin/activate
.base:
	. .venv/bin/activate && pip install -U pip setuptools wheel
.main:
	. .venv/bin/activate && pip install torch transformers
	. .venv/bin/activate && pip install -r requirements.txt

.extras:
	. .venv/bin/activate && pip install -U isort black ruff pytest pytest-cov

install: .venv  .base .main .extras


# Linters

.isort:
	. .venv/bin/activate && isort ${SOURCES} ${TESTS}

.black:
	. .venv/bin/activate && black ${SOURCES} ${TESTS} 

.ruff:
	. .venv/bin/activate && ruff check --fix ${SOURCES} ${TESTS}

.assets:
	test -d dataset || mkdir dataset
	test -s dataset/ml-1m.zip  || wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-1m.zip -O dataset/ml-1m.zip
	test -d dataset/ml-1m  || unzip dataset/ml-1m.zip -d dataset/

.pytest_s3:
	. .venv/bin/activate && pytest -v -v  ${TESTS} --cov=${SOURCES} --cov-report=xml --capture=no -k TestS3Train

.pytest:
	. .venv/bin/activate && pytest -v -v  ${TESTS} --cov=${SOURCES} --cov-report=xml --capture=no 

.lint: .isort .black .ruff
lint: .venv .lint

.test: .assets .extras .pytest
test: .test

test_s3: .extras .pytest_s3

build: 
	rm -f dist/*
	. .venv/bin/activate && pip install build && python -m build .
	. .venv/bin/activate && pip install twine && twine upload dist/de4rec*

# Cleaning

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf ${VENV}
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	rm -rf dataset/*

reinstall: clean install
