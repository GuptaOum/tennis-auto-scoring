"""Publish the fine-tuned ball detector to the Hugging Face Hub.

    hf auth login                      # once, with a WRITE token
    python training/upload_to_hf.py    # add --private to keep it unlisted

Only ``models/ball_finetuned.pt`` is uploaded. The other two files in
``models/`` are the upstream project's weights, not ours, and republishing
someone else's trained artefacts under our account would be wrong regardless of
whether the licence permits it.

Authentication is deliberately left to ``hf auth login`` rather than accepting a
token argument: a token passed on the command line lands in shell history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "models" / "ball_finetuned.pt"
CARD = REPO_ROOT / "models" / "MODEL_CARD.md"
DEFAULT_NAME = "tennis-ball-detector-yolov8m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default=DEFAULT_NAME, help=f"repo name (default: {DEFAULT_NAME})"
    )
    parser.add_argument(
        "--owner", default=None, help="account or org (default: the logged-in user)"
    )
    parser.add_argument("--private", action="store_true", help="create it unlisted")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    if not WEIGHTS.exists():
        print(f"missing weights: {WEIGHTS}", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        identity = api.whoami()
    except Exception:
        print(
            "not logged in. Run `hf auth login` with a WRITE token first "
            "(https://huggingface.co/settings/tokens).",
            file=sys.stderr,
        )
        return 1

    owner = args.owner or identity["name"]
    repo_id = f"{owner}/{args.name}"
    print(f"uploading to {repo_id} ({'private' if args.private else 'public'})")

    try:
        api.create_repo(repo_id, repo_type="model", private=args.private,
                        exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(WEIGHTS),
            path_in_repo="ball_finetuned.pt",
            repo_id=repo_id,
            commit_message="fine-tuned tennis ball detector, YOLOv8m @ 960px",
        )
        if CARD.exists():
            api.upload_file(
                path_or_fileobj=str(CARD),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="model card",
            )
    except HfHubHTTPError as exc:
        # The overwhelmingly common cause is a read-only token.
        print(f"upload failed: {exc}", file=sys.stderr)
        print(
            "If this is a 403, the token is read-only - create one with write "
            "access at https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    print(f"done: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
