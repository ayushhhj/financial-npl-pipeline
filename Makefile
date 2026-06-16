.PHONY: up down test lint format

up:
	docker compose up -d

down:
	docker compose down

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	black src/ tests/

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_trf

pipeline:
	python -m src.ingestion.edgar
	python -m src.nlp.ner
	python -m src.nlp.embeddings
	python -m src.nlp.topics
	python -m src.graph.loader