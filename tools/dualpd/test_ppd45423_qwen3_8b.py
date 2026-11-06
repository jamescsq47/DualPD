"""GPU validation for SGLang PR #45423 using the local Qwen3-8B model."""

import math
import random
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)


MODEL = "/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3-8B"
CHUNK_SIZE = 128


class TestPPD45423Qwen3_8B(PDDisaggregationServerBase):
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        "--disaggregation-decode-enable-local-prefill",
        "--chunked-prefill-size",
        str(CHUNK_SIZE),
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = MODEL
        cls.launch_all()
        cls.tokenizer = get_tokenizer(cls.model)

    def _token_ids(self, input_len, num_prompts):
        return [
            list(request.prompt)
            for request in sample_random_requests(
                input_len=input_len,
                output_len=1,
                num_prompts=num_prompts,
                range_ratio=1.0,
                tokenizer=self.tokenizer,
                dataset_path="",
                return_text=False,
            )
        ]

    def _generate(self, url, input_ids, max_new_tokens, **fields):
        response = requests.post(
            f"{url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                **fields,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def _pd(self, input_ids, max_new_tokens, **fields):
        return self._generate(self.lb_url, input_ids, max_new_tokens, **fields)

    def _local(self, input_ids, max_new_tokens):
        return self._generate(
            self.decode_url,
            input_ids,
            max_new_tokens,
            disagg_local_prefill=True,
        )

    @staticmethod
    def _first_token_kl(reference, actual):
        p = {
            token[1]: token[0]
            for token in reference["meta_info"]["output_top_logprobs"][0]
        }
        q = {
            token[1]: token[0]
            for token in actual["meta_info"]["output_top_logprobs"][0]
        }
        return sum(math.exp(logp) * (logp - q.get(token, -30.0)) for token, logp in p.items())

    def test_decode_local_prefill_reuses_turn_one_cache(self):
        history_len, suffix_len, output_len = 1024, 3 * CHUNK_SIZE + 17, 32
        histories = self._token_ids(history_len, 4)
        suffixes = self._token_ids(suffix_len, 4)

        for history, suffix in zip(histories, suffixes):
            turn_one = self._pd(history, output_len)
            prompt = history + turn_one["output_ids"] + suffix
            local = self._local(prompt, output_len)
            reference = self._pd(prompt, output_len, cache_salt=uuid.uuid4().hex)

            assert local["meta_info"]["cached_tokens"] >= history_len
            assert len(local["output_ids"]) == output_len
            assert self._first_token_kl(reference, local) < 0.05

    def test_local_prefill_and_regular_pd_can_share_decode(self):
        rng = random.Random(0)
        histories = self._token_ids(512, 8)
        for history in histories:
            self._pd(history, 8)
        suffixes = self._token_ids(4 * CHUNK_SIZE, 32)

        def run(index):
            prompt = histories[index % len(histories)] + suffixes[index]
            output_len = rng.choice([1, 16, 64])
            if index % 2:
                return self._local(prompt, output_len)
            return self._pd(prompt, output_len)

        with ThreadPoolExecutor(16) as pool:
            outputs = list(pool.map(run, range(len(suffixes))))

        assert all(
            output["meta_info"]["finish_reason"]["type"] == "length"
            for output in outputs
        )
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)

    def test_prefill_rejects_decode_local_request(self):
        response = requests.post(
            f"{self.prefill_url}/generate",
            json={
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_new_tokens": 1},
                "disagg_local_prefill": True,
            },
            timeout=60,
        )
        assert response.status_code == 400
        assert "--disaggregation-decode-enable-local-prefill" in response.text
