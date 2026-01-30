import json
import unittest
from unittest.mock import MagicMock, patch

import aihelp


class DummyEmbedder(aihelp.EmbeddingBackend):
    def embed(self, texts):
        vectors = []
        for t in texts:
            if t.startswith("query:"):
                vectors.append([1.0, 0.0])
            elif "first" in t:
                vectors.append([0.9, 0.1])
            elif "second" in t:
                vectors.append([0.1, 0.9])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class AiHelpTests(unittest.TestCase):
    def test_default_embedder_singleton_used_when_none_passed(self):
        help_text = "first chunk text\n\nsecond chunk text"

        class CountingEmbedder(aihelp.EmbeddingBackend):
            def __init__(self):
                self.calls = 0

            def embed(self, texts):
                self.calls += 1
                return [[1.0, 0.0]] * len(texts)

        stub = CountingEmbedder()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("aihelp.core.capture_help_text", return_value=help_text), patch("aihelp.core._get_default_embedder", return_value=stub):
            aihelp.aihelp(lambda x: x, "q1", add_documentation=True, openai_client=mock_client)
            aihelp.aihelp(lambda x: x, "q2", add_documentation=True, openai_client=mock_client)

        self.assertEqual(stub.calls, 2, "default embedder should be reused across calls")

    def test_popular_library_extras_handles_getattr_errors(self):
        class BadAttrs:
            def __getattr__(self, name):
                if name in {"shape", "dtype", "ndim", "size"}:
                    raise RuntimeError("no details")
                raise AttributeError

        hdr = aihelp.object_header(BadAttrs())
        self.assertIn("repr", hdr)

    def test_aihelp_survives_help_failure(self):
        class NoHelp:
            pass

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("aihelp.core.capture_help_text", side_effect=RuntimeError("boom")):
            result = aihelp.aihelp(NoHelp(), "q", add_documentation=True, openai_client=mock_client)

        self.assertEqual(result, "ok")
        _, kwargs = mock_client.responses.create.call_args
        payload = json.loads(kwargs["input"])
        self.assertNotIn("documentation_chunks", payload)

    def test_add_documentation_ranks_and_attaches_top_chunk(self):
        help_text = "first chunk text\n\nsecond chunk text"
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("aihelp.core.capture_help_text", return_value=help_text):
            result = aihelp.aihelp(
                lambda x: x,
                "question about first",
                add_documentation=True,
                top_k_docs=1,
                chunk_chars=16,
                embedder=DummyEmbedder(),
                openai_client=mock_client,
            )

        self.assertEqual(result, "ok")
        _, kwargs = mock_client.responses.create.call_args
        payload = json.loads(kwargs["input"])
        self.assertIn("documentation_chunks", payload)
        self.assertEqual(payload["documentation_chunks"], ["first chunk text"])

    def test_no_add_documentation_skips_help_capture(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("aihelp.core.capture_help_text") as cap_help:
            result = aihelp.aihelp(
                123,
                "plain question",
                add_documentation=False,
                openai_client=mock_client,
            )

        self.assertEqual(result, "ok")
        cap_help.assert_not_called()


if __name__ == "__main__":
    unittest.main()
