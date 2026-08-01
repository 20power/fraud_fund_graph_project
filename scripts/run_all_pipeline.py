from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "01_prepare_data.py",
    "02_build_features.py",
    "03_train_evaluate.py",
    "04_link_analysis.py",
    "05_robustness_experiments.py",
    "06_generate_reports.py",
    "07_extended_stability.py",
    "08_enhance_cases.py",
    "09_prepare_ui_data.py",
]


def main() -> None:
    log_path = ROOT / "logs" / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for stage in STAGES:
            message = f"[{datetime.now():%F %T}] START {stage}"
            print(message)
            log.write(message + "\n")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / stage)], cwd=ROOT,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            )
            print(result.stdout, end="")
            log.write(result.stdout)
            if result.returncode:
                raise SystemExit(f"阶段失败：{stage}；日志：{log_path}")
            done = f"[{datetime.now():%F %T}] PASS {stage}"
            print(done)
            log.write(done + "\n")
        verify = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_delivery.py")], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
        )
        print(verify.stdout, end="")
        log.write(verify.stdout)
        if verify.returncode:
            raise SystemExit(f"交付检查失败；日志：{log_path}")
    print(f"完整流程结束。日志：{log_path}")


if __name__ == "__main__":
    main()
