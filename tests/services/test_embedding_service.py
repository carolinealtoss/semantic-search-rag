import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from tests.services.support import isolated_service


class EmbeddingServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = self.enterContext(
            isolated_service("services.embedding_service")
        )

    def test_load_documents_splits_files_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "primeiro.txt").write_text("a" * 501, encoding="utf-8")
            (path / "segundo.txt").write_text("Olá", encoding="utf-8")
            (path / "ignorado.md").write_text("ignorado", encoding="utf-8")
            with patch.object(self.service, "DOCUMENTS_PATH", path):
                documents = self.service.load_documents()

        by_id = {document["id"]: document for document in documents}
        self.assertEqual(set(by_id), {"primeiro_0", "primeiro_1", "segundo_0"})
        self.assertEqual(by_id["primeiro_0"]["content"], "a" * 500)
        self.assertEqual(by_id["primeiro_1"]["content"], "a" * 101)
        self.assertEqual(by_id["primeiro_1"]["chunk_index"], 1)
        self.assertEqual(by_id["segundo_0"]["filename"], "segundo.txt")
        self.assertEqual(by_id["segundo_0"]["content"], "Olá")

    def test_load_documents_returns_empty_for_no_text_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.service, "DOCUMENTS_PATH", Path(directory)):
                self.assertEqual(self.service.load_documents(), [])

    def test_load_documents_propagates_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "invalido.txt").write_bytes(b"\xff")
            with patch.object(self.service, "DOCUMENTS_PATH", path):
                with self.assertRaises(UnicodeDecodeError):
                    self.service.load_documents()

    def test_generate_embeddings_encodes_texts_as_float32(self):
        model = Mock()
        model.encode.return_value = np.array([[0.5, 0.25]], dtype=np.float64)
        with patch.object(self.service, "MODEL", model):
            embeddings = self.service.generate_embeddings([{"content": "texto"}])

        model.encode.assert_called_once_with(
            ["texto"], convert_to_numpy=True, normalize_embeddings=True
        )
        self.assertEqual(embeddings.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(embeddings, [[0.5, 0.25]])

    def test_generate_embeddings_rejects_empty_documents(self):
        model = Mock()
        with patch.object(self.service, "MODEL", model):
            with self.assertRaisesRegex(ValueError, "Nenhum documento encontrado"):
                self.service.generate_embeddings([])
        model.encode.assert_not_called()

    def test_generate_embeddings_propagates_model_failure(self):
        model = Mock()
        model.encode.side_effect = RuntimeError("erro no modelo")
        with patch.object(self.service, "MODEL", model):
            with self.assertRaisesRegex(RuntimeError, "erro no modelo"):
                self.service.generate_embeddings([{"content": "texto"}])

    def test_split_text_handles_overlap_and_paragraphs(self):
        self.assertEqual(
            self.service.split_text("abcdef", chunk_size=3, overlap=1),
            ["abc", "cde", "ef"],
        )
        self.assertEqual(
            self.service.split_text("abc\n\ndef", chunk_size=6, overlap=1),
            ["abc", "c\n\ndef"],
        )
        self.assertEqual(self.service.split_text(" \n\n "), [])

    def test_split_text_rejects_invalid_chunk_parameters(self):
        for chunk_size, overlap in ((0, 0), (-1, 0), (3, -1), (3, 3), (3, 4)):
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    self.service.split_text("texto", chunk_size, overlap)
