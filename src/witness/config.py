"""YAML configuration loading with one machine-specific root.

Every path in the configs is written relative to ${WITNESS_ROOT}, the directory that holds the raw corpora, the model
checkpoints and the experiment outputs (layout in README.md).  The variable must be set; an unresolved reference is
an error, never a silent default.
"""
import os
import re

import yaml

_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _expand(node):
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    if isinstance(node, str):
        return _VAR.sub(lambda m: os.environ[m.group(1)], node)
    return node


def load_config(path):
    """Parse a YAML file and substitute ${VAR} references from the environment (KeyError if a variable is unset)."""
    with open(path, encoding="utf8") as fh:
        return _expand(yaml.safe_load(fh))
