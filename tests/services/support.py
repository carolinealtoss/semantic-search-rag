import importlib
import sys
from contextlib import contextmanager
from unittest.mock import patch

import services


@contextmanager
def isolated_service(name):
    module_names = ("embedding_service", "search_service")
    missing = object()
    previous_modules = {
        f"services.{item}": sys.modules.get(f"services.{item}", missing)
        for item in module_names
    }
    previous_attributes = {
        item: getattr(services, item, missing)
        for item in module_names
    }

    try:
        # A inicialização real pode carregar ou baixar o modelo durante o import.
        with patch("sentence_transformers.SentenceTransformer"):
            yield importlib.import_module(name)
    finally:
        for module_name, previous in previous_modules.items():
            if previous is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        for attribute, previous in previous_attributes.items():
            if previous is missing:
                if hasattr(services, attribute):
                    delattr(services, attribute)
            else:
                setattr(services, attribute, previous)
