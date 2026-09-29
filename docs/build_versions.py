#!/usr/bin/env python3
"""Build versioned Moto documentation into one GitHub Pages artifact."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


TAG_PATTERN = re.compile(r"^v(\d+)\.(\d+)(?:\.(\d+))?$")
REQUIRED_PATHS = (
    "docs/site/conf.py",
    "docs/stubs/moto/__init__.pyi",
    "docs/Doxyfile.pages",
)
VERSION_CONFIG = r"""

# Injected into this temporary worktree by docs/build_versions.py.
import json as _moto_json
import os as _moto_os

templates_path = list(globals().get("templates_path", []) or [])
if "_templates" not in templates_path:
    templates_path.append("_templates")
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "versioning.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}
html_context = dict(globals().get("html_context", {}) or {})
html_context.update({
    "moto_doc_versions": _moto_json.loads(_moto_os.environ["MOTO_DOC_VERSIONS"]),
    "moto_doc_current": _moto_os.environ["MOTO_DOC_CURRENT"],
    "moto_doc_current_url": _moto_os.environ["MOTO_DOC_CURRENT_URL"],
})
html_title = "Moto Documentation — " + _moto_os.environ["MOTO_DOC_CURRENT_LABEL"]
html_theme_options = dict(globals().get("html_theme_options", {}) or {})
html_theme_options["source_branch"] = _moto_os.environ["MOTO_DOC_SOURCE_REF"]
html_theme_options["announcement"] = (
    "Documentation version: <strong>"
    + _moto_os.environ["MOTO_DOC_CURRENT_LABEL"]
    + "</strong>"
)
html_baseurl = _moto_os.environ["MOTO_DOC_BASE_URL"]
"""


def run(
    command: list[str],
    cwd: Path,
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def resolve(repository: Path, *refs: str) -> str | None:
    for ref in refs:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=repository,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def has_docs(repository: Path, commit: str) -> bool:
    return all(
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
        for path in REQUIRED_PATHS
    )


def version_tuple(tag: str) -> tuple[int, int, int]:
    match = TAG_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError(f"invalid semantic tag {tag!r}")
    return tuple(int(part or 0) for part in match.groups())


def discover_refs(
    repository: Path, branches: list[str], minimum_tag: str
) -> list[dict]:
    refs: list[dict] = []
    for branch in branches:
        commit = resolve(
            repository,
            f"refs/remotes/origin/{branch}",
            f"refs/heads/{branch}",
        )
        if commit is None or not has_docs(repository, commit):
            raise RuntimeError(f"cannot build documentation branch {branch!r}")
        label = "main (stable)" if branch == "main" else f"{branch} (development)"
        refs.append(
            {"name": branch, "label": label, "commit": commit, "source": branch}
        )

    minimum = version_tuple(minimum_tag)
    tags = run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/tags"],
        repository,
        capture=True,
    ).splitlines()
    releases = []
    for tag in tags:
        if not TAG_PATTERN.fullmatch(tag) or version_tuple(tag) < minimum:
            continue
        commit = resolve(repository, f"refs/tags/{tag}")
        if commit is not None and has_docs(repository, commit):
            releases.append(
                {
                    "name": tag,
                    "label": tag,
                    "commit": commit,
                    "source": tag,
                    "version": version_tuple(tag),
                }
            )
    releases.sort(key=lambda ref: ref["version"], reverse=True)
    for ref in releases:
        ref.pop("version")
    return refs + releases


def configure_worktree(worktree: Path) -> None:
    site = worktree / "docs/site"
    templates = site / "_templates"
    templates.mkdir(exist_ok=True)
    shutil.copy2(
        Path(__file__).parent / "versioning/versioning.html",
        templates / "versioning.html",
    )
    with (site / "conf.py").open("a", encoding="utf-8") as stream:
        stream.write(VERSION_CONFIG)


def build_ref(
    repository: Path,
    temporary: Path,
    output: Path,
    ref: dict,
    versions: list[dict],
    base_path: str,
) -> None:
    worktree = temporary / ref["name"]
    destination = output / ref["name"]
    print(f"Building {ref['label']} from {ref['commit'][:12]}", flush=True)
    run(
        ["git", "worktree", "add", "--detach", "--force", str(worktree), ref["commit"]],
        repository,
    )
    try:
        configure_worktree(worktree)
        current_url = f"{base_path}{ref['name']}/"
        environment = os.environ.copy()
        environment.update(
            {
                "MOTO_DOC_VERSIONS": json.dumps(versions),
                "MOTO_DOC_CURRENT": ref["name"],
                "MOTO_DOC_CURRENT_LABEL": ref["label"],
                "MOTO_DOC_CURRENT_URL": current_url,
                "MOTO_DOC_SOURCE_REF": ref["source"],
                "MOTO_DOC_BASE_URL": f"https://haizhouz.github.io{current_url}",
            }
        )
        run(
            [
                "sphinx-build",
                "-W",
                "--keep-going",
                "-b",
                "html",
                "docs/site",
                str(destination),
            ],
            worktree,
            env=environment,
        )

        warnings = output.parent / "doxygen-warnings"
        warnings.mkdir(parents=True, exist_ok=True)
        config = temporary / f"Doxyfile.{ref['name']}"
        config.write_text(
            f"@INCLUDE = {worktree / 'docs/Doxyfile.pages'}\n"
            f"OUTPUT_DIRECTORY = {destination / 'cpp'}\n"
            f"PROJECT_NUMBER = {ref['label']}\n"
            f"WARN_LOGFILE = {warnings / (ref['name'] + '.log')}\n",
            encoding="utf-8",
        )
        run(["doxygen", str(config)], worktree)
    finally:
        run(["git", "worktree", "remove", "--force", str(worktree)], repository)


def redirect_page(target: str) -> str:
    escaped = html.escape(target, quote=True)
    encoded = json.dumps(target)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={escaped}">'
        f'<link rel="canonical" href="{escaped}">'
        "<title>Moto documentation</title></head><body>"
        f'<a href="{escaped}">Open Moto documentation</a>'
        "<script>location.replace("
        f"{encoded} + location.search + location.hash"
        ");</script>"
        "</body></html>\n"
    )


def write_legacy_redirects(output: Path, default_ref: str, base_path: str) -> None:
    default_root = output / default_ref
    for source in default_root.rglob("*.html"):
        relative = source.relative_to(default_root)
        if relative.parts[0] == "cpp" and relative != Path("cpp/index.html"):
            continue
        destination = output / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            redirect_page(f"{base_path}{default_ref}/{relative.as_posix()}"),
            encoding="utf-8",
        )


def write_root(
    output: Path, versions: list[dict], default_ref: str, base_path: str
) -> None:
    target = f"{base_path}{default_ref}/"
    output.joinpath("index.html").write_text(redirect_page(target), encoding="utf-8")
    output.joinpath("versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    write_legacy_redirects(output, default_ref, base_path)
    output.joinpath(".nojekyll").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/_build/html"))
    parser.add_argument("--base-path", default="/moto/")
    parser.add_argument("--branches", nargs="+", default=["main", "dev"])
    parser.add_argument("--minimum-tag", default="v2.3")
    parser.add_argument("--default-ref", default="main")
    arguments = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    output = (repository / arguments.output).resolve()
    if output in (repository, repository.parent) or len(output.parts) < 3:
        raise RuntimeError(f"unsafe documentation output path {output}")
    base_path = f"/{arguments.base_path.strip('/')}/"
    refs = discover_refs(repository, arguments.branches, arguments.minimum_tag)
    if arguments.default_ref not in {ref["name"] for ref in refs}:
        raise RuntimeError(f"unpublished default ref {arguments.default_ref!r}")
    versions = [
        {
            "name": ref["name"],
            "label": ref["label"],
            "url": f"{base_path}{ref['name']}/",
            "commit": ref["commit"],
        }
        for ref in refs
    ]

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="moto-docs-") as path:
        try:
            for ref in refs:
                build_ref(repository, Path(path), output, ref, versions, base_path)
        finally:
            run(["git", "worktree", "prune"], repository)
    write_root(output, versions, arguments.default_ref, base_path)
    print(f"Versioned documentation written to {output}")


if __name__ == "__main__":
    main()
