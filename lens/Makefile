.PHONY: install test lint fmt typecheck api web demo clean all

all: lint typecheck test

install:
	pip install -e ".[api,dev]"
	cd web && npm install

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src
	cd web && npm run typecheck

api:
	uvicorn lens.api:app --reload --port 8000

web:
	cd web && npm run dev

demo:
	./scripts/demo.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
