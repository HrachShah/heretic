# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    disable_progress_bars,
    enable_progress_bars,
)
from requests.exceptions import RequestException

from .utils import print


def collect_reproducibles(path: str):
    print(
        f"Collecting [bold]reproduce.json[/] files from Hugging Face and storing them in [bold]{path}[/]..."
    )
    print()

    api = HfApi()

    models = api.list_models(
        filter=["heretic", "reproducible"],
        sort="created_at",
    )

    found = 0
    downloaded = 0
    skipped = 0

    # We're only downloading tiny files, so the progress bars are just noise.
    disable_progress_bars()

    try:
        for model in models:
            # Ignore repositories containing quantizations.
            if model.tags is not None and "gguf" in model.tags:
                continue

            print(f"[bold]{model.id}[/]... ", end="")

            try:
                user, repository = model.id.split("/")
            except ValueError:
                # Hugging Face model IDs are always "user/repo". An entry
                # that doesn't split into exactly two parts is malformed
                # and not something this collector can store; skip it
                # rather than crashing the whole scan.
                print(f"[yellow]skipped[/] (malformed model id: {model.id!r})")
                skipped += 1
                continue

            try:
                paths_info = api.get_paths_info(
                    model.id,
                    "reproduce/reproduce.json",
                    expand=True,
                )
            except (RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError, RequestException) as error:
                # Repository/entry not found is a normal "no reproduce.json" case
                # at the API level; an HTTP error or connection drop is a
                # transient failure we want to skip past instead of losing all
                # already-collected files.
                print(f"[yellow]skipped[/] ({type(error).__name__}: {error})")
                skipped += 1
                continue

            # The reproduce.json file might not exist in the repository
            # despite the relevant tags being present.
            if not paths_info:
                print("[yellow]no reproduce.json found[/]")
                continue

            found += 1

            commit_hash = paths_info[0].last_commit.oid

            file_path = (
                Path(path)
                / "huggingface.co"
                / user
                / f"{repository}-{commit_hash[:7]}.json"
            )
            if file_path.exists():
                print(" already stored")
                continue

            try:
                cache_path = hf_hub_download(
                    model.id,
                    "reproduce/reproduce.json",
                )
            except (RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError, RequestException) as error:
                # Path can vanish between the listing call and the actual
                # download (a 404 here, a rate-limit 429, or a network blip
                # on a long-running scan over hundreds of repos). Don't let
                # one bad repo abort the whole collection.
                print(f"[yellow]skipped[/] ({type(error).__name__}: {error})")
                skipped += 1
                continue

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache_path, file_path)
            except OSError as error:
                # Local filesystem failure (disk full, permission denied,
                # cache file vanished between download and copy). The
                # download itself succeeded, but we couldn't persist it,
                # so decrement `found` to keep the "Found" count honest
                # about what the user actually has on disk.
                print(f"[yellow]skipped[/] ({type(error).__name__}: {error})")
                skipped += 1
                found -= 1
                continue

            print(" [green]downloaded[/]")

            downloaded += 1
    finally:
        enable_progress_bars()

    print()
    print(f"Found: [bold]{found}[/] files")
    print(f"Downloaded: [bold]{downloaded}[/] files")
    print(f"Already stored: [bold]{found - downloaded}[/] files")
    if skipped:
        print(f"Skipped: [bold]{skipped}[/] repositories (see messages above)")
