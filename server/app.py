"""Shim — re-exports from patronet.app for openenv CLI compatibility."""

from patronet.app import app


def main(host: str = "0.0.0.0", port: int = 8000):
  import uvicorn

  uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
  main()
