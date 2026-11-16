"""Live Ollama (`qwen3.5:4b`) GPU Kernel & SRAM Cost-Model CEGIS Audit."""

from aether_compiler.audit.ollama_compiler import (
    build_compiler_incident_suite,
    run_real_compiler_ollama_audit,
)

__all__ = ["build_compiler_incident_suite", "run_real_compiler_ollama_audit"]
