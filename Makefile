.PHONY: up down seed migrate test eval lint typecheck fmt

up:  ## build and start app, postgres and langfuse
	test -f .env || cp .env.example .env
	docker compose up -d --build --wait

down:
	docker compose down

migrate:
	docker compose exec app alembic upgrade head

seed:  ## load seed/*.json into the running stack
	docker compose exec app python -m app.seed

test:  ## needs Docker for testcontainers, no API keys
	uv run pytest

eval:  ## run the eval set against DATABASE_URL (re-seeds the demo restaurants); ARGS=--judge
	uv run python -m evals.run $(ARGS)

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy app tests evals

fmt:
	uv run ruff format .
	uv run ruff check --fix .
