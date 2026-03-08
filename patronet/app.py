"""FastAPI application for the Patronet Emergency Environment."""

from openenv.core.env_server.http_server import create_app

from patronet.models import PatronetAction, PatronetObservation
from patronet.environment import PatronetEnvironment

app = create_app(
  PatronetEnvironment,
  PatronetAction,
  PatronetObservation,
  env_name="patronet_emergency",
  max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
  import uvicorn

  uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
  main()
