import json
import unittest
from unittest.mock import MagicMock, patch

import helpit


class DummyEmbedder(helpit.EmbeddingBackend):
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


class HelpitTests(unittest.TestCase):
    def tearDown(self):
        helpit.set_default_client(None)

    def test_raises_when_no_client_and_no_default(self):
        with self.assertRaises(ValueError):
            helpit.helpit(123, "needs a client", openai_client=None)

    def test_uses_default_client_when_set(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp
        helpit.set_default_client(mock_client)

        with patch("builtins.print"):
            result = helpit.helpit(123, "plain question", add_documentation=False, echo=True)

        self.assertEqual(result, "ok")
        mock_client.responses.create.assert_called_once()

    def test_default_embedder_singleton_used_when_none_passed(self):
        help_text = "first chunk text\n\nsecond chunk text"

        class CountingEmbedder(helpit.EmbeddingBackend):
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

        with patch("helpit.core.capture_help_text", return_value=help_text), patch("helpit.core._get_default_embedder", return_value=stub):
            helpit.helpit(lambda x: x, "q1", add_documentation=True, openai_client=mock_client, echo=False)
            helpit.helpit(lambda x: x, "q2", add_documentation=True, openai_client=mock_client, echo=False)

        self.assertEqual(stub.calls, 2, "default embedder should be reused across calls")

    def test_popular_library_extras_handles_getattr_errors(self):
        class BadAttrs:
            def __getattr__(self, name):
                if name in {"shape", "dtype", "ndim", "size"}:
                    raise RuntimeError("no details")
                raise AttributeError

        hdr = helpit.object_header(BadAttrs())
        self.assertIn("repr", hdr)

    def test_helpit_survives_help_failure(self):
        class NoHelp:
            pass

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("helpit.core.capture_help_text", side_effect=RuntimeError("boom")):
            result = helpit.helpit(NoHelp(), "q", add_documentation=True, openai_client=mock_client, echo=False)

        self.assertIsNone(result)
        _, kwargs = mock_client.responses.create.call_args
        payload = json.loads(kwargs["input"])
        self.assertNotIn("documentation_chunks", payload)

    def test_add_documentation_ranks_and_attaches_top_chunk(self):
        help_text = "first chunk text\n\nsecond chunk text"
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("helpit.core.capture_help_text", return_value=help_text):
            result = helpit.helpit(
                lambda x: x,
                "question about first",
                add_documentation=True,
                top_k_docs=1,
                chunk_chars=16,
                embedder=DummyEmbedder(),
                openai_client=mock_client,
                echo=False,
            )

        self.assertIsNone(result)
        _, kwargs = mock_client.responses.create.call_args
        payload = json.loads(kwargs["input"])
        self.assertIn("documentation_chunks", payload)
        self.assertEqual(payload["documentation_chunks"], ["first chunk text"])

    def test_no_add_documentation_skips_help_capture(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("helpit.core.capture_help_text") as cap_help:
            result = helpit.helpit(
                123,
                "plain question",
                add_documentation=False,
                openai_client=mock_client,
                echo=False,
            )

        self.assertIsNone(result)
        cap_help.assert_not_called()

    def test_echo_prints_and_returns_text(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("builtins.print") as mock_print:
            result = helpit.helpit(
                123,
                "plain question",
                add_documentation=False,
                openai_client=mock_client,
                echo=True,
            )

        self.assertEqual(result, "ok")
        mock_print.assert_called_once_with("ok")

    def test_default_echo_prints(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("builtins.print") as mock_print:
            result = helpit.helpit(
                123,
                "plain question",
                add_documentation=False,
                openai_client=mock_client,
            )

        self.assertIsNone(result)
        mock_print.assert_called_once_with("ok")

    def test_echo_false_prints_and_returns_none(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "ok"
        mock_client.responses.create.return_value = mock_resp

        with patch("builtins.print") as mock_print:
            result = helpit.helpit(
                123,
                "plain question",
                add_documentation=False,
                openai_client=mock_client,
                echo=False,
            )

        self.assertIsNone(result)
        mock_print.assert_called_once_with("ok")


if __name__ == "__main__":
    unittest.main()
