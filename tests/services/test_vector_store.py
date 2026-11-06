import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services import vector_store


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.storage = Path(directory) / "storage"
        self.index_path = self.storage / "documents.index"
        self.metadata_path = self.storage / "metadata.json"
        self.enterContext(patch.object(vector_store, "STORAGE_PATH", self.storage))
        self.enterContext(patch.object(vector_store, "INDEX_PATH", self.index_path))
        self.enterContext(patch.object(vector_store, "METADATA_PATH", self.metadata_path))
        self.documents = [
            {"id": "doc_0", "filename": "doc.txt", "chunk_index": 0, "content": "Olá"}
        ]
        self.embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    def test_create_and_load_persist_index_and_metadata(self):
        created = vector_store.create_vector_store(self.documents, self.embeddings)
        loaded, documents = vector_store.load_vector_store()

        self.assertEqual(created.ntotal, 1)
        self.assertEqual(loaded.ntotal, 1)
        self.assertEqual(documents, self.documents)
        np.testing.assert_array_equal(loaded.reconstruct(0), self.embeddings[0])

    def test_create_propagates_index_write_failure(self):
        with patch.object(vector_store.faiss, "write_index", side_effect=RuntimeError("falha")):
            with self.assertRaisesRegex(RuntimeError, "falha"):
                vector_store.create_vector_store(self.documents, self.embeddings)

    def test_load_propagates_missing_index(self):
        with self.assertRaises(RuntimeError):
            vector_store.load_vector_store()

    def test_load_propagates_missing_metadata(self):
        vector_store.create_vector_store(self.documents, self.embeddings)
        self.metadata_path.unlink()
        with self.assertRaises(FileNotFoundError):
            vector_store.load_vector_store()

    def test_load_propagates_invalid_metadata(self):
        vector_store.create_vector_store(self.documents, self.embeddings)
        self.metadata_path.write_text("{", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            vector_store.load_vector_store()
