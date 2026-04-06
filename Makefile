.PHONY: install run serve report clean

install:
	pip install -e ".[dev]"

run:
	python scripts/run_all.py --config configs/experiments.yaml --results-dir results

run-quick:
	python scripts/run_all.py --config configs/experiments.yaml --results-dir results \
		--only baseline lcm_8step deepcache_2 compiled preset_balanced --n-runs 1

serve:
	python server/serve.py --preset balanced --port 8000

serve-speed:
	python server/serve.py --preset speed --port 8000

report:
	python scripts/generate_report.py

lint:
	ruff check src/ scripts/ server/

clean:
	rm -rf results/images results/*.csv results/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
