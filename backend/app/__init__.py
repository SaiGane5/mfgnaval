"""Package initializer for the `app` package.

This exposes the ASGI application under two attribute names so different
uvicorn invocation styles work:

- `uvicorn app.main:app` (module `app.main`, attribute `app`) — preferred
- `uvicorn app:main` (module `app`, attribute `main`) — also supported

We import the FastAPI app instance defined in `app.main` and re-export it as
both `app` and `main` for compatibility.
"""
from .main import app as app  # re-export the FastAPI application

# Also provide `main` alias so commands like `uvicorn app:main` resolve correctly
main = app

__all__ = ["app", "main"]
