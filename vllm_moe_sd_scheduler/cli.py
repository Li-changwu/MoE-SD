import argparse
import json

from .config import SchedulerConfig
from .entrypoints import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True, help="scheduler config as JSON string")
    args = parser.parse_args()

    payload = json.loads(args.config_json)
    cfg = SchedulerConfig.from_dict(payload)
    binding = build_runtime(cfg.feature_flags)

    print(json.dumps({"mode": binding.mode, "reason": binding.reason}, ensure_ascii=False))


if __name__ == "__main__":
    main()
