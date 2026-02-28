# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Sidecar delivery runtime for Coriro.

Serialization and injection of ColorMeasurement data into VLM contexts.
Supports multiple delivery mechanisms:

1. Tool Output -- For tool-use / function-calling models
2. System Prompt -- Injection into system instructions
3. Context Block -- Structured block in conversation context

The delivery layer never modifies measurement content.
"""

from coriro.runtime.serializers import (
    SerializerFormat,
    BlockFormat,
    to_context_block,
    to_system_prompt,
    to_tool_output,
)

__all__ = [
    "to_tool_output",
    "to_system_prompt",
    "to_context_block",
    "SerializerFormat",
    "BlockFormat",
]

