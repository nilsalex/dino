-include .env
export

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

run-xvfb:
	Xvfb $(DISPLAY_NAME) -screen 0 $(BROWSER_WIDTH)x$(BROWSER_HEIGHT)x24

view-xvfb:
	DISPLAY=$(DISPLAY_NAME) gst-launch-1.0 ximagesrc ! videoconvert ! autovideosink

run-chromium:
	DISPLAY=$(DISPLAY_NAME) WAYLAND_DISPLAY= chromium \
		--ozone-platform=x11 \
		--user-data-dir=/tmp/chrome-headless \
		--no-first-run \
		--window-size=$(BROWSER_WIDTH),$(BROWSER_HEIGHT) \
		--app=$(BROWSER_URL)

run-headless:
	DISPLAY=$(DISPLAY_NAME) HEADLESS=true uv run python -m src.train_local

view-headless:
	gst-launch-1.0 udpsrc port=$(UDP_PORT) ! jpegdec ! videoconvert ! autovideosink

view-agent:
	gst-launch-1.0 udpsrc port=$(UDP_PORT_AGENT) ! jpegdec ! videoconvert ! autovideosink

run-local:
	uv run python -m src.train_local
