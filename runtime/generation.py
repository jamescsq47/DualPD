"""Tokenizer and sampling state for inference, without the training rollout."""
import asyncio

from .processing_utils import load_processor, load_tokenizer


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def clear_instances(cls):
        cls._instances = {}


class GenerateState(metaclass=SingletonMeta):
    def __init__(self, args):
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus
            // args.rollout_num_gpus_per_engine
        )
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        if getattr(args, "sglang_enable_deterministic_inference", False):
            self.group_sampling_seeds = [
                args.rollout_seed + i for i in range(args.n_samples_per_prompt)
            ]
        self.abort_epoch = 0
        self.aborted = False
