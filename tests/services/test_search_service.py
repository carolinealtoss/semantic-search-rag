import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np

from tests.services.support import isolated_service


class SearchServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = self.enterContext(
            isolated_service("services.search_service")
        )
        self.model = Mock()
        self.model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float64)
        self.enterContext(patch.object(self.service, "MODEL", self.model))

    def test_search_returns_ranked_results_and_limits_top_k(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[0.5, 0.0], [1.0, 0.0]], dtype=np.float32))
        documents = [
            {"id": "a_0", "filename": "a.txt", "chunk_index": 0, "content": "A"},
            {"id": "b_1", "filename": "b.txt", "chunk_index": 1, "content": "B"},
        ]

        results = self.service.search_documents("consulta", index, documents, top_k=5)

        self.model.encode.assert_called_once_with(
            ["consulta"], convert_to_numpy=True, normalize_embeddings=True
        )
        self.assertEqual(results, [
            {"id": "b_1", "filename": "b.txt", "chunk_index": 1,
             "content": "B", "similarity": 1.0},
            {"id": "a_0", "filename": "a.txt", "chunk_index": 0,
             "content": "A", "similarity": 0.5},
        ])

    def test_search_skips_missing_faiss_indices(self):
        index = Mock(ntotal=1)
        index.search.return_value = (
            np.array([[0.0]], dtype=np.float32),
            np.array([[-1]], dtype=np.int64),
        )
        self.assertEqual(self.service.search_documents("consulta", index, []), [])

    def test_search_propagates_embedding_failure(self):
        self.model.encode.side_effect = RuntimeError("erro no modelo")
        index = Mock()
        with self.assertRaisesRegex(RuntimeError, "erro no modelo"):
            self.service.search_documents("consulta", index, [])
        index.search.assert_not_called()

    def test_search_propagates_index_failure(self):
        index = Mock(ntotal=1)
        index.search.side_effect = RuntimeError("erro no índice")
        with self.assertRaisesRegex(RuntimeError, "erro no índice"):
            self.service.search_documents("consulta", index, [])

    def test_search_propagates_missing_document_metadata(self):
        index = Mock(ntotal=1)
        index.search.return_value = (
            np.array([[1.0]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        with self.assertRaises(IndexError):
            self.service.search_documents("consulta", index, [])
