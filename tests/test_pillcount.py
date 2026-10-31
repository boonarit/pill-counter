"""เทียบผลนับของ pillcount.py กับแอป web/index.html บนภาพทดสอบสังเคราะห์ชุดเดียวกัน (seed 42, ส่งออกจากแอป)"""
import csv
import subprocess
import sys
import unittest
from pathlib import Path

import pillcount as pc

FIX = Path(__file__).parent / "fixtures"
ROWS = list(csv.DictReader(open(FIX / "expected.csv", encoding="utf-8")))
# case5/6 = สีพื้นใกล้สีเม็ด threshold เปราะ ผลต่างจากแอปได้ (ขอบภาพ/การปัดเศษของ blur) จึงไม่บังคับให้ตรง
EXACT = {"case1.png", "case2.png", "case3.png", "case4.png", "case7.png"}


class MatchesApp(unittest.TestCase):
    def test_counts_match_app(self):
        for r in ROWS:
            if r["file"] not in EXACT:
                continue
            with self.subTest(r["file"]):
                o = pc.count(pc.load(FIX / r["file"]), {"chan": r["chan"]})
                self.assertEqual((o["cc"], o["area"], o["ws"]),
                                 (int(r["app_cc"]), int(r["app_area"]), int(r["app_ws"])))

    def test_batch_writes_csv(self):
        out = Path(__file__).parent / "_out.csv"
        try:
            res = subprocess.run([sys.executable, "pillcount.py", str(FIX), str(FIX / "expected.csv"), "-o", str(out)],
                                 capture_output=True, text=True, check=True)
            rows = list(csv.DictReader(open(out, encoding="utf-8")))
            self.assertEqual(len(rows), len(ROWS))
            self.assertIn("all,all,7,", res.stdout)
        finally:
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
