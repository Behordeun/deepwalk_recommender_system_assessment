# ----------------------------------------------------------------------------
# 🌍 Auto-detect .env
# ----------------------------------------------------------------------------

ENV_PATH := $(shell find src -maxdepth 1 -name ".env" 2>/dev/null)
-include $(ENV_PATH)
export

# ----------------------------------------------------------------------------
# 🐍 Python Version Handling
# ----------------------------------------------------------------------------

PYTHON_VERSION ?= $(shell python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
USE_PYENV ?= true
ifeq ($(USE_PYENV), true)
	PYTHON_BIN := $(HOME)/.pyenv/versions/$(PYTHON_VERSION)/bin/python
else
	PYTHON_BIN := python$(PYTHON_VERSION)
endif

# ----------------------------------------------------------------------------
# 💻 OS Detection (for venv activation compatibility)
# ----------------------------------------------------------------------------

OS := $(shell uname)
ifeq ($(OS),Linux)
	ACTIVATE_VENV = source .venv/bin/activate
else ifeq ($(OS),Darwin)
	ACTIVATE_VENV = source .venv/bin/activate
else
	ACTIVATE_VENV = .venv\\Scripts\\activate
endif

# ----------------------------------------------------------------------------
# 👞 Dependency Management
# ----------------------------------------------------------------------------

.PHONY: install install-fast venv venv-dev ensure-uv activate-venv auto-activate bootstrap

ensure-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "🚀 Installing uv..."; \
		if [ "$(OS)" = "Darwin" ]; then \
			brew install astral-sh/uv/uv || pipx install uv; \
		else \
			curl -Ls https://astral.sh/uv/install.sh | bash; \
		fi \
	}

install: ensure-uv
	@echo "📆 Installing dependencies with Poetry..."
	poetry install

install-fast: ensure-uv venv
	@echo "⚡ Exporting dependencies..."
	poetry export -f requirements.txt --without-hashes -o requirements.txt
	uv pip install -r requirements.txt --python ".venv/bin/python"

venv: ensure-uv
	@echo "🧱 Creating virtual environment (.venv) using uv with Python $(PYTHON_VERSION)..."
	uv venv ".venv" --python "$(PYTHON_BIN)"
	".venv/bin/python" -m ensurepip --upgrade
	@echo "📆 Exporting and installing dependencies with uv..."
	poetry export -f requirements.txt --without-hashes -o requirements.txt
	uv pip install -r requirements.txt --python ".venv/bin/python"
	@echo "✅ Virtual environment is ready."

venv-dev: ensure-uv
	@echo "🧱 Creating dev virtual environment (.venv) using uv with Python $(PYTHON_VERSION)..."
	uv venv ".venv" --python "$(PYTHON_BIN)"
	@echo "📆 Exporting and installing dev dependencies with uv..."
	poetry export --dev -f requirements.txt --without-hashes -o dev-requirements.txt
	uv pip install -r dev-requirements.txt --python ".venv/bin/python"
	@echo "✅ Dev environment ready."

activate-venv:
	@echo "🐍 Activating virtual environment..."
	@$(ACTIVATE_VENV); exec $$SHELL

auto-activate:
	@echo "🔍 Checking for .venv..."
	@if [ -d .venv ]; then \
		echo "🐍 Found .venv. Activating..."; \
		$(ACTIVATE_VENV); exec $$SHELL; \
	else \
		echo "❌ .venv not found. Run `make venv` first."; \
	fi

bootstrap: ensure-uv
	@echo "🧱 Creating virtual environment using uv..."
	uv venv ".venv" --python "$(PYTHON_BIN)"
	poetry export -f requirements.txt --without-hashes -o requirements.txt
	uv pip install -r requirements.txt --python ".venv/bin/python"
	@if [ ! -f src/.env ] && [ -f src/.env.example ]; then \
		echo "📋 Copying .env.example to .env..."; \
		cp src/.env.example src/.env; \
	fi
	@echo "🔍 Validating .env file..."
	".venv/bin/python" -c "from config import settings; print('✅ .env validation passed.')"
	@if [ -f .pre-commit-config.yaml ]; then \
		echo "🧩 Detected .pre-commit-config.yaml — installing pre-commit..."; \
		".venv/bin/pip" install pre-commit && \
		".venv/bin/pre-commit" install && \
		echo "✅ pre-commit hooks activated."; \
	else \
		echo "ℹ️ No .pre-commit-config.yaml found. Skipping pre-commit setup."; \
	fi
	@make format
	@make lint
	@make test
	@echo "🎉 Bootstrap complete."
	@echo "👉 Now run: source .venv/bin/activate"

# ----------------------------------------------------------------------------
# 🪥 Formatting & Linting
# ----------------------------------------------------------------------------

.PHONY: format lint clean

format:
	@echo "🔧 Running autoflake..."
	autoflake --remove-unused-variables --remove-all-unused-imports -ri .
	@echo "📆 Running isort..."
	isort .
	@echo "🎨 Running black..."
	black .

lint:
	@echo "🔎 Running flake8..."
	flake8 .

clean:
	@echo "🪝 Cleaning __pycache__ and .pyc files..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# ----------------------------------------------------------------------------
# 🧪 Testing with Coverage
# ----------------------------------------------------------------------------

.PHONY: test

test:
	@echo "🧪 Running tests with coverage..."
	@mkdir -p html
	".venv/bin/python" -m pytest --tb=short -q \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:html

# ----------------------------------------------------------------------------
# 🚀 Development
# ----------------------------------------------------------------------------

.PHONY: dev

dev:
	@echo "🔄 Starting dev server..."
	poetry run uvicorn src.deepwalk_recommender.main:app \
		--host 0.0.0.0 --port 8000 \
		--reload \
		--reload-dir src/deepwalk_recommender \
		--reload-exclude .venv

# ----------------------------------------------------------------------------
# 🚀 Production
# ----------------------------------------------------------------------------

.PHONY: prod

prod:
	@echo "🔄 Starting prod server..."
	poetry run uvicorn src.deepwalk_recommender.main:app \
		--host 0.0.0.0 --port 8000

# ----------------------------------------------------------------------------
# 🐳 Docker
# ----------------------------------------------------------------------------

.PHONY: docker-build run-docker

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t herbehordeun/deepwalk_recommender:latest .

run-docker:
	@echo "🐳 Starting full dev stack using Docker Compose..."
	docker compose up -d

# ----------------------------------------------------------------------------
# ✅ .env Validation (standalone)
# ----------------------------------------------------------------------------

.PHONY: check-env

check-env:
	@echo "🔍 Checking environment configuration..."
	poetry run python -c "from src.deepwalk_recommender.api.api_server:app.config.app_config import env_config; print('✅ .env validation passed.')"

# ----------------------------------------------------------------------------
# 🚀 Versioning & Release
# ----------------------------------------------------------------------------

.PHONY: release

release:
	@echo "🚀 Bumping version and tagging release..."
	@read -p "Enter release type (patch, minor, major): " type; \
	".venv/bin/pip" install bump2version && \
	bump2version $$type && \
	git push && \
	git push --tags && \
	echo "✅ Release bumped and tagged."

# ----------------------------------------------------------------------------
# 🥺 Environment Diagnostics
# ----------------------------------------------------------------------------

.PHONY: doctor

doctor:
	@echo "🥺 Running environment diagnostics..."
	@if [ ! -d ".venv" ]; then \
		echo "❌ No .venv found. Run 'make venv' or 'make bootstrap' first."; \
		exit 1; \
	fi
	@VENV_PY=$$(realpath ".venv/bin/python"); \
	CURRENT_PY=$$(which python); \
	if [ "$$CURRENT_PY" != "$$VENV_PY" ]; then \
		echo "⚠️  Python is NOT from the virtual environment."; \
		echo "    Current: $$CURRENT_PY"; \
		echo "    Expected: $$VENV_PY"; \
		echo "💡 Run: source .venv/bin/activate"; \
	else \
		echo "✅ Python is correctly using the virtual environment."; \
	fi
	@VENV_PIP=$$(realpath ".venv/bin/pip"); \
	CURRENT_PIP=$$(which pip); \
	if [ "$$CURRENT_PIP" != "$$VENV_PIP" ]; then \
		echo "⚠️  pip is NOT from the virtual environment."; \
		echo "    Current: $$CURRENT_PIP"; \
		echo "    Expected: $$VENV_PIP"; \
		echo "💡 Run: source .venv/bin/activate"; \
	else \
		echo "✅ pip is correctly using the virtual environment."; \
	fi
	@echo "🔍 Checking if required packages are installed..."
	@".venv/bin/python" -c "import fastapi, uvicorn, pydantic; print('✅ Required packages detected.')"
	@echo "🧪 Virtual environment health check complete."
