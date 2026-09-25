PY ?= .venv/bin/python

install:
	python3.11 -m venv .venv
	@if [ "$$(uname)" = Darwin ] && ! $(PY) -c "import pybullet" 2>/dev/null; then $(MAKE) pybullet-mac; fi
	$(PY) -m pip install -r requirements.txt
	@cp -n .env.example .env 2>/dev/null || true
# PyPI has no macOS arm64 pybullet wheel, and the source build fails on the macOS 26 SDK because the
# bundled zlib defines fdopen() as a macro. Build 3.2.7 with that classic-Mac-only stub removed.
pybullet-mac:
	rm -rf build/pybullet && mkdir -p build/pybullet
	$(PY) -m pip download pybullet==3.2.7 --no-binary :all: --no-deps -q -d build/pybullet
	cd build/pybullet && tar xzf pybullet-3.2.7.tar.gz && \
		sed -i.bak '/^#ifndef fdopen$$/,/^#endif$$/d' pybullet-3.2.7/examples/ThirdPartyLibs/zlib/zutil.h
	$(PY) -m pip install build/pybullet/pybullet-3.2.7
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
trace:
	$(PY) scripts/analyze_trace.py

.PHONY: install pybullet-mac run fake test lint urdf poses sweep render trace
