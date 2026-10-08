PY ?= .venv/bin/python

.PHONY: setup scrape parse label train score signals backtest all test

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

scrape parse label train score signals backtest:
	$(PY) -m fintrax $@

all:
	$(PY) -m fintrax all

test:
	$(PY) -m pytest -q
