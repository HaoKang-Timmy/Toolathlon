import unittest
import json
import os
import tempfile

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, CompletionUsage
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.responses import Response

from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

from utils.api_model.model_provider import (
    OpenAIChatCompletionsModelWithRetry,
    _normalize_tool_arguments_for_history,
)


class FakeChatCompletionStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk


class OpenAIChatCompletionsModelWithRetryTests(unittest.IsolatedAsyncioTestCase):
    def test_normalize_tool_arguments_for_history(self) -> None:
        self.assertEqual(
            json.loads(_normalize_tool_arguments_for_history('{"city":"Paris"}')),
            {"city": "Paris"},
        )
        self.assertEqual(
            json.loads(_normalize_tool_arguments_for_history('{"city":"Paris"')),
            {"city": "Paris"},
        )
        self.assertEqual(
            json.loads(
                _normalize_tool_arguments_for_history(
                    '{"query":"q","variables":{"project":"p"}'
                )
            ),
            {"query": "q", "variables": {"project": "p"}},
        )
        self.assertEqual(
            json.loads(_normalize_tool_arguments_for_history('{"city":"Paris]')),
            {"_raw_tool_arguments": '{"city":"Paris]'},
        )
        self.assertEqual(
            json.loads(_normalize_tool_arguments_for_history('["Paris"]')),
            {"_raw_tool_arguments": ["Paris"]},
        )

    def build_model(self) -> OpenAIChatCompletionsModelWithRetry:
        client = AsyncOpenAI(api_key="test-key", base_url="https://example.com/v1")
        return OpenAIChatCompletionsModelWithRetry(
            model="test-model",
            openai_client=client,
            retry_times=1,
            retry_delay=0.0,
            debug=False,
        )

    async def test_raw_get_response_keeps_standard_chat_completion_path(self) -> None:
        model = self.build_model()

        chat_completion = ChatCompletion(
            id="chatcmpl_1",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(role="assistant", content="hello"),
                )
            ],
            created=0,
            model="test-model",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

        async def fake_fetch_response(*args, **kwargs):
            return chat_completion

        model._fetch_response = fake_fetch_response  # type: ignore[method-assign]

        response = await model.raw_get_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
        )

        self.assertEqual(response.usage.input_tokens, 3)
        self.assertEqual(response.usage.output_tokens, 2)
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(response.output[0].type, "message")
        self.assertEqual(response.output[0].content[0].text, "hello")

    async def test_raw_get_response_handles_tuple_stream_fallback(self) -> None:
        model = self.build_model()

        initial_response = Response(
            id="resp_1",
            created_at=0.0,
            model="test-model",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )
        stream = FakeChatCompletionStream(
            [
                ChatCompletionChunk(
                    id="chatcmpl_1",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content="hello streamed", role="assistant"),
                            finish_reason=None,
                            index=0,
                            logprobs=None,
                        )
                    ],
                    created=0,
                    model="test-model",
                    object="chat.completion.chunk",
                    usage=None,
                ),
                ChatCompletionChunk(
                    id="chatcmpl_1",
                    choices=[],
                    created=0,
                    model="test-model",
                    object="chat.completion.chunk",
                    usage=CompletionUsage(prompt_tokens=7, completion_tokens=4, total_tokens=11),
                ),
            ]
        )

        async def fake_fetch_response(*args, **kwargs):
            return initial_response, stream

        model._fetch_response = fake_fetch_response  # type: ignore[method-assign]

        response = await model.raw_get_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
        )

        self.assertEqual(response.usage.input_tokens, 7)
        self.assertEqual(response.usage.output_tokens, 4)
        self.assertEqual(response.usage.total_tokens, 11)
        self.assertEqual(response.output[0].type, "message")
        self.assertEqual(response.output[0].content[0].text, "hello streamed")

    async def test_request_metrics_capture_stream_latency_and_tokens(self) -> None:
        model = self.build_model()
        initial_response = Response(
            id="resp_metrics", created_at=0.0, model="test-model", object="response",
            output=[], tool_choice="auto", tools=[], parallel_tool_calls=False,
        )
        stream = FakeChatCompletionStream([
            ChatCompletionChunk(
                id="chunk_1", choices=[ChunkChoice(delta=ChoiceDelta(content="hello", role="assistant"), finish_reason=None, index=0, logprobs=None)],
                created=0, model="test-model", object="chat.completion.chunk", usage=None,
            ),
            ChatCompletionChunk(
                id="chunk_2", choices=[], created=0, model="test-model", object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12),
            ),
        ])

        async def fake_fetch_response(*args, **kwargs):
            self.assertTrue(kwargs["stream"])
            return initial_response, stream

        model._fetch_response = fake_fetch_response  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "requests.jsonl")
            old = os.environ.get("TOOLATHLON_REQUEST_METRICS_PATH")
            os.environ["TOOLATHLON_REQUEST_METRICS_PATH"] = path
            try:
                await model.get_response(
                    system_instructions=None, input="hi", model_settings=ModelSettings(), tools=[],
                    output_schema=None, handoffs=[], tracing=ModelTracing.DISABLED, previous_response_id=None,
                )
            finally:
                if old is None:
                    os.environ.pop("TOOLATHLON_REQUEST_METRICS_PATH", None)
                else:
                    os.environ["TOOLATHLON_REQUEST_METRICS_PATH"] = old
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["prompt_tokens"], 9)
            self.assertEqual(record["decode_tokens"], 3)
            self.assertIsNotNone(record["ttft_ms"])
            self.assertIsNotNone(record["tpot_ms"])


if __name__ == "__main__":
    unittest.main()
