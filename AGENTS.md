# DualPD development

This independent checkout is the maintained extraction target explicitly
requested by the user. The original slime tree is a frozen comparison source.
Read src/dualpd/docs/AGENTIC_PD_DESIGN_INVARIANTS.md before lifecycle changes.
Keep snapshot ownership, fences, TP decisions, sampling, harness contracts and
measurement windows unchanged during extraction. Independent audit and relevant
lifecycle/fault tests are required before GPU validation. CPU checks do not
constitute final throughput acceptance (300 s warmup + 1200 s measurement).
