import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prefetch_swe_oci as p
import load_swe_oci_into_docker as loader


class PrefetchSweOciTests(unittest.TestCase):
    def test_image_name_matches_harness_convention(self):
        row = {"instance_id": "astropy__astropy-12907"}
        self.assertEqual(
            p.image_name(row),
            "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        )
        self.assertEqual(p.image_name(dict(row, image_name="registry/image:tag")),
                         "registry/image:tag")

    def test_atomic_json_never_leaves_temporary_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            p.atomic_json(path, {"completed": {"x": "image"}})
            self.assertEqual(json.loads(path.read_text())["completed"]["x"], "image")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_docker_import_uses_harness_repository_name(self):
        source = "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        self.assertEqual(loader.docker_target(source), source.removeprefix("docker.io/"))


if __name__ == "__main__":
    unittest.main()
