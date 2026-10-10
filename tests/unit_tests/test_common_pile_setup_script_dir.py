# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSetupScriptDirectory(unittest.TestCase):
    def test_relative_invocation_finds_adjacent_dataset_script(self):
        source = Path("tools/common_pile_dataset/setup_common_pile_dataset.sh").read_text()
        # Run the actual directory-discovery phase, excluding HPC prerequisites
        # and the subsequent network/download/preprocessing phases.
        phase = source.split("# Create work directory", 1)[1].split("# Clone Megatron-LM", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            dataset_script = scripts / "create_common_pile_ci_dataset.py"
            dataset_script.touch()
            setup = scripts / "setup.sh"
            setup.write_text(
                "#!/bin/bash\nset -euo pipefail\n"
                f"WORK_DIR='{root / 'work'}'\n" + phase + '\nprintf "%s\\n" "${DATASET_SCRIPT}"\n'
            )
            result = subprocess.run(
                ["bash", "scripts/setup.sh"], cwd=root, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                Path(result.stdout.splitlines()[-1]).resolve(), dataset_script.resolve()
            )


if __name__ == "__main__":
    unittest.main()
