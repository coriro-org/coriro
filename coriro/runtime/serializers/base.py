# Copyright (c) 2026 Coriro
# SPDX-License-Identifier: MIT

"""Base types and utilities for serializers."""

from enum import Enum


class SerializerFormat(Enum):
    """Output format for serializers."""

    JSON = "json"
    JSON_PRETTY = "json_pretty"
    NATURAL = "natural"

