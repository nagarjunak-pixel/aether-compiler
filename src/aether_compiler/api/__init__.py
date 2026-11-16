"""FastAPI application for AetherCompiler MLIR/Triton Graph Compiler & Polyhedral Workbench."""

from aether_compiler.api.app import create_app

__all__ = ["create_app"]
