import subprocess
import unittest
import os

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMINE_FILE_PATH = os.path.join(EXAMPLES_DIR, "../../bin/examine.py")


class ExamineTest(unittest.TestCase):
    def test_examine_env(self):
        envs = [
            "env_oasis.py"
        ]
        for env in envs:
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.check_call(
                    ["/usr/bin/env", "python", EXAMINE_FILE_PATH, os.path.join(EXAMPLES_DIR, env)],
                    timeout=10)
