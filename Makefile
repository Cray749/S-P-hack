.PHONY: setup test lint run docker data eval backtest
PY ?= python

setup:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install --no-deps -e .
test:             ## fast tests (light mode, no model downloads)
	$(PY) -m pytest
lint:
	$(PY) -m ruff check src tests scripts
run:              ## start the API + dashboard on :8000
	$(PY) -m tremor.api
docker:           ## build and start the container
	docker compose up --build
data:             ## rebuild data from public sources (needs network)
	$(PY) scripts/fetch_market.py && $(PY) scripts/build_analogues.py && $(PY) scripts/build_corpus.py && $(PY) -m tremor.book
backtest:         ## out-of-sample backtest of the analogue model -> data/results/backtest.json
	$(PY) -m tremor.backtest
eval:             ## NLP + event-study evaluation -> data/results/eval.json
	$(PY) -m tremor.evaluate
