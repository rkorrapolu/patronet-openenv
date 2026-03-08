FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY patronet/ patronet/
COPY data/ data/

RUN pip install --no-cache-dir -e .

EXPOSE 9696

ENTRYPOINT ["uvicorn", "patronet.env:app"]
CMD ["--host", "0.0.0.0", "--port", "9696"]
