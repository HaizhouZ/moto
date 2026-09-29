# Releasing Moto

`VERSION` is the authoritative project version. It uses three-component
semantic versions without a `v` prefix, for example `2.3.0`. CMake, the Python
extension, and the installed CMake package version are derived from this file.

Git tags and GitHub releases add the prefix. For compatibility with the
existing release convention, a zero patch component is omitted from the tag:

```text
VERSION: 2.3.0
tag:     v2.3
release: Moto 2.3.0
```

## Upstream release

1. Update `VERSION` on `dev` and push the release commit.
2. Wait for the `Build` workflow on that exact commit to pass.
3. Open GitHub Actions, select `Release`, choose the `dev` branch, and run the
   workflow. It reads `VERSION`, creates the tag, and publishes the GitHub
   release with generated notes.

The workflow refuses to publish when the matching build has not passed or the
tag/release already exists. Do not retarget or replace a published tag; release
a new patch version instead.

## conda-forge release

The conda package is named `libmoto`; the installed Python import remains
`moto`. The initial recipe is submitted to `conda-forge/staged-recipes`. After
acceptance, conda-forge creates `libmoto-feedstock` and its version-update bot
tracks Moto's GitHub releases.

For a new upstream version, the feedstock recipe version changes and its build
number resets to zero. Recipe-only fixes keep the upstream version and
increment the build number. Review and merge the bot's update after its build,
import, runtime-codegen, and CMake-consumer tests pass.

The first conda-forge release targets Linux. macOS and Windows require a
portable replacement for the POSIX runtime compilation, locking, dynamic
library naming, and loading paths before those platforms can be enabled.

## Versioned documentation

The documentation workflow publishes `main`, `dev`, and semantic release tags
starting at `v2.3` into one GitHub Pages artifact. Each version's tutorials,
Python API reference, and Doxygen C++ reference are built from the same commit.
The site root redirects to `main`, and every version provides a selector for
the other published refs.

The workflow requires a full Git checkout because `docs/build_versions.py`
uses temporary detached worktrees. A release tag is published automatically
when it contains the current documentation tree; feature branches are not
published.
