#!/usr/bin/env python3
"""Collector must not accept truncated/pending/failed array accounting."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect_b1 import accounting

class AccountingTests(unittest.TestCase):
    def test_requires_each_completed_array_element(self):
        good="10|COMPLETED|0:0\n11_0|COMPLETED|0:0\n11_1|COMPLETED|0:0\n"
        want=["10","11_0","11_1"]
        self.assertEqual(len(accounting(good,want)),3)
        for bad in (good.replace("11_1|COMPLETED|0:0", "11_1|FAILED|1:0"),
                    good.replace("11_1|COMPLETED|0:0\n", ""),
                    good.replace("11_1|COMPLETED|0:0", "11_[1]|PENDING|0:0")):
            with self.assertRaises(RuntimeError): accounting(bad,want)

if __name__ == "__main__": unittest.main()
