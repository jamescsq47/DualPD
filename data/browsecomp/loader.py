"""Load BrowseComp while preserving its message-list prompt."""

from __future__ import annotations

import logging

from dualpd.runtime.dataset import Dataset
from dualpd.runtime.types import Sample

from dualpd.runtime.workloads.api import LoadContext
from dualpd.runtime.workloads.config import DatasetSpec


LOG = logging.getLogger(__name__)


def load_samples(context: LoadContext, dataset: DatasetSpec) -> list[Sample]:
    options = dataset.options
    source = Dataset(
        dataset.path,
        tokenizer=context.tokenizer,
        processor=context.processor,
        max_length=context.args.rollout_max_prompt_len,
        prompt_key=str(options.get("prompt_key", context.args.input_key)),
        multimodal_keys=options.get("multimodal_keys", context.args.multimodal_keys),
        label_key=options.get("label_key", context.args.label_key),
        metadata_key=str(options.get("metadata_key", context.args.metadata_key)),
        tool_key=options.get("tool_key", context.args.tool_key),
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        seed=context.args.rollout_seed,
    )
    LOG.info("Loaded %d BrowseComp samples from %s", len(source.origin_samples), dataset.path)
    return source.origin_samples
