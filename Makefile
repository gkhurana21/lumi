PY ?= .venv/bin/python

install:
	python3.11 -m venv .venv && $(PY) -m pip install -r requirements.txt && cp -n .env.example .env || true
run:
	$(PY) -m uvicorn app.main:app --port 8000
fake:
	PROVIDERS=fake $(PY) -m uvicorn app.main:app --port 8000
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check app tests scripts
urdf:
	$(PY) scripts/inspect_urdf.py
poses:
	$(PY) scripts/sim_poses.py
sweep:
	$(PY) scripts/sim_poses.py --sweep
render:
	$(PY) scripts/sim_poses.py --render out/poses

.PHONY: install run fake test lint urdf poses sweep render
