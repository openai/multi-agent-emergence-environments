import subprocess
import unittest
import pytest
import os

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMINE_FILE_PATH = os.path.join(EXAMPLES_DIR, "../bin/examine.py")

class ExamineTest(unittest.TestCase):
    def test_examine_env(self):
        envs = [
            "hide_and_seek_full.jsonnet",
            "hide_and_seek_quadrant.jsonnet",
            "blueprint.jsonnet",
            "lock_and_return.jsonnet",
            "sequential_lock.jsonnet",
            "shelter.jsonnet",
        ]
        for env in envs:
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.check_call(
                    ["/usr/bin/env", "python", EXAMINE_FILE_PATH, os.path.join(EXAMPLES_DIR, env)],
                    timeout=10)


    def test_examine_policies(self):
        envs_policies = [
            ("hide_and_seek_full.jsonnet", "hide_and_seek_full.npz"),
            ("hide_and_seek_quadrant.jsonnet", "hide_and_seek_quadrant.npz"),
            ("blueprint.jsonnet", "blueprint.npz"),
            ("lock_and_return.jsonnet", "lock_and_return.npz"),
            ("sequential_lock.jsonnet", "sequential_lock.npz"),
            ("shelter.jsonnet", "shelter.npz"),
        ]
        for env, policy in envs_policies:
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.check_call(
                    ["/usr/bin/env", "python", EXAMINE_FILE_PATH, os.path.join(EXAMPLES_DIR, env), os.path.join(EXAMPLES_DIR, policy)],
                    timeout=15)
