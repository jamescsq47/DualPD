import copy
import json
import math
from pathlib import Path
import unittest

from dram_complete_frame_analysis import complete_frames, P_TOKEN, D_TOKEN, P_GPU, D_GPU


ENV = {"PREFILL_PORTS": "27300 27320 27400 27420", "DECODE_PORTS": "27301 27321 27401 27421"}


def fixture():
    rows = []
    for ts in (0, 60, 115, 120):
        for role, port_key, token, gpu in (("prefill", "PREFILL_PORTS", P_TOKEN, P_GPU),
                                           ("decode", "DECODE_PORTS", D_TOKEN, D_GPU)):
            endpoints = [dict(endpoint=f"http://127.0.0.1:{port}",
                              metrics={token: 100000 + ts*10 + i, gpu: 100+ts})
                         for i, port in enumerate(ENV[port_key].split())]
            rows.append(dict(ts=ts, role=role, endpoint_metrics=endpoints,
                             metrics={key: sum(x["metrics"][key] for x in endpoints) for key in (token,gpu)}))
    return rows


class CompleteFrameTests(unittest.TestCase):
    def test_missing_last_endpoint_cannot_create_false_counter_drop(self):
        rows = fixture()
        rows[-1]["endpoint_metrics"].pop()
        for key in (D_TOKEN, D_GPU):
            rows[-1]["metrics"][key] = sum(x["metrics"][key] for x in rows[-1]["endpoint_metrics"])
        original = copy.deepcopy(rows)
        frames, audit = complete_frames(rows, ENV, 0, 120)
        self.assertLess(rows[-1]["metrics"][D_TOKEN], rows[1]["metrics"][D_TOKEN])
        self.assertEqual(audit["tail_120s_decode_tokens"], 4600)
        self.assertEqual(frames[-1][0], 115)
        self.assertEqual(audit["measurement_complete_frames"], 3)
        self.assertEqual(rows, original)

    def test_real_stall_still_fails(self):
        rows = fixture()
        for row in rows:
            if row["role"] == "decode":
                for endpoint in row["endpoint_metrics"]:
                    endpoint["metrics"][D_TOKEN] = 100000
                row["metrics"][D_TOKEN] = 400000
        with self.assertRaisesRegex(RuntimeError, "no token progress"):
            complete_frames(rows, ENV, 0, 120)

    def test_endpoint_reset_is_not_hidden_by_other_endpoint_growth(self):
        rows = fixture()
        row = rows[3]
        row["endpoint_metrics"][0]["metrics"][D_TOKEN] = 1
        row["endpoint_metrics"][1]["metrics"][D_TOKEN] += 200000
        row["metrics"][D_TOKEN] = sum(x["metrics"][D_TOKEN] for x in row["endpoint_metrics"])
        with self.assertRaisesRegex(RuntimeError, "counter reset"):
            complete_frames(rows, ENV, 0, 120)

    def test_incomplete_window_edges_fail_closed(self):
        rows = fixture()[:4]
        with self.assertRaisesRegex(RuntimeError, "measurement edges"):
            complete_frames(rows, ENV, 0, 120)

    def test_reset_in_rejected_partial_frame_is_still_detected(self):
        rows = fixture()
        row = rows[3]
        row["endpoint_metrics"].pop()
        row["endpoint_metrics"][0]["metrics"][D_TOKEN] = 1
        with self.assertRaisesRegex(RuntimeError, "counter reset"):
            complete_frames(rows, ENV, 0, 120)

    def test_inconsistent_aggregate_is_rejected(self):
        rows = fixture(); rows[-1]["metrics"][D_TOKEN] += 1
        with self.assertRaisesRegex(RuntimeError, "aggregate does not match"):
            complete_frames(rows, ENV, 0, 120)

    def test_original_four_results_are_numerically_unchanged(self):
        root = Path('/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022')
        states = json.loads((root/'state.json').read_text())
        for value in states.values():
            directory = Path(value['run_dir'])
            boundary = json.loads((directory/'closed_loop_boundaries.json').read_text())
            env = json.loads((directory/'launch_config.json').read_text())['env']
            with (directory/'engine_metrics.jsonl').open() as stream:
                rows = [json.loads(x) for x in stream]
            frames, audit = complete_frames(rows, env, boundary['measurement_start_wall'], boundary['measurement_end_wall'])
            selected = [(ts,f) for ts,f in frames if boundary['measurement_start_wall']<=ts<=boundary['measurement_end_wall']]
            delta = selected[-1][1]['decode']['metrics'][D_TOKEN]-selected[0][1]['decode']['metrics'][D_TOKEN]
            self.assertTrue(math.isclose(delta/audit['counter_span_seconds'],value['result']['decode_tps'],rel_tol=1e-12))


if __name__ == '__main__':
    unittest.main()
