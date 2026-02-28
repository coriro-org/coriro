# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""
Serializers for ColorMeasurement delivery to VLMs.

Each serializer formats a ColorMeasurement for a specific injection method.
All serializers preserve the measurement exactly -- no modification or inference.
"""

from coriro.runtime.serializers.base import SerializerFormat
from coriro.runtime.serializers.block import to_context_block, BlockFormat
from coriro.runtime.serializers.system import to_system_prompt
from coriro.runtime.serializers.tool import to_tool_output

__all__ = [
    "SerializerFormat",
    "BlockFormat",
    "to_tool_output",
    "to_system_prompt",
    "to_context_block",
]

