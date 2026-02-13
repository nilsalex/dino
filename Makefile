install:
	uv sync

lock:
	uv lock

lint:
	uv run mypy src/
	uv run ruff format --check src/
	uv run ruff check src/

format:
	uv run ruff format src/
	uv run ruff check --fix src/

run:
	uv run python -m src.main

run-headless:
	HEADLESS=true UDP_PORT=5000 UDP_PORT_AGENT=5001 GAME=reaction CROP_X=500 CROP_Y=170 CROP_WIDTH=340 CROP_HEIGHT=400 uv run python -m src.train_local

view-headless:
	gst-launch-1.0 udpsrc port=5000 ! jpegdec ! videoconvert ! autovideosink

view-agent:
	gst-launch-1.0 udpsrc port=5001 ! jpegdec ! videoconvert ! autovideosink

run-local:
	uv run python -m src.train_local
