# Releasing Moto

`VERSION` is the authoritative project version. It uses three-component
semantic versions without a `v` prefix, for example `2.3.0`. CMake, the Python
extension, and the installed CMake package version are derived from this file.

Git tags and GitHub releases add the prefix:

```text
VERSION: 2.3.0
tag:     v2.3.0
release: v2.3.0
```

## Upstream release

1. Update `VERSION` on `dev` and run the Release build and test suite.
2. Merge the release commit to `main`.
3. Tag the exact `main` commit and push the tag:

   ```bash
   version="$(tr -d '[:space:]' < VERSION)"
   git tag -a "v${version}" -m "Moto ${version}"
   git push origin main "v${version}"
   ```

4. Create the GitHub release from that tag. Do not retarget or replace a
   published tag; release a new patch version instead.

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
