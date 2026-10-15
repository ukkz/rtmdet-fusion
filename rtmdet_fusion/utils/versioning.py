"""Version embedding and compatibility checks.

Versioning rules (see AGENTS.md "Versioning"):
- The only version axis is the git tag (SemVer, `vX.Y.Z`).
- A model is trained from a clean commit that already carries its version tag, so the version
  embedded in the checkpoint (`git describe --tags --dirty`) is exactly that tag.
  tools/train.py refuses to start otherwise, unless the run is marked with --experiment.
- Exported artifacts (ONNX / CoreML / tfjs) inherit the checkpoint's version, i.e. the code
  that produced the weights, not the HEAD at export time.
- Loaders warn about MAJOR mismatches (MAJOR = checkpoint-incompatible change).
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

# Repository root (two levels above rtmdet_fusion/utils/versioning.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]

_RELEASE_RE = re.compile(r'^v\d+\.\d+\.\d+$')


def get_code_version() -> Optional[str]:
    """Return the version string of the current code.

    Result of `git describe --tags --dirty --always`,
    e.g. 'v2.0.0', 'v2.0.0-3-gec14e78', 'v2.0.0-3-gec14e78-dirty'.
    None when git is unavailable (e.g. a distributed source tree).
    """
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--dirty', '--always'],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def is_release_version(version: Optional[str]) -> bool:
    """True if the version is exactly a release tag (clean checkout of a `vX.Y.Z` tag)."""
    return bool(version) and _RELEASE_RE.match(version) is not None


def parse_major(version: Optional[str]) -> Optional[int]:
    """Extract the MAJOR number from a version string (e.g. 'v2.1.0-3-gabc' -> 2)."""
    if not version:
        return None
    m = re.match(r'^v?(\d+)\.', version)
    return int(m.group(1)) if m else None


def check_checkpoint_version(checkpoint: dict, tool_name: str = 'loader') -> Optional[str]:
    """Check the version embedded in a checkpoint and print warnings.

    - Warns when the checkpoint is not a release (experimental run or uncommitted changes)
    - Warns when its MAJOR differs from the current code (state_dict may be incompatible)

    Returns:
        The embedded version string, or None if the checkpoint has none.
    """
    ckpt_version = checkpoint.get('version')
    if ckpt_version is None:
        print(f"Warning: [{tool_name}] Checkpoint has no embedded version.")
        return None

    print(f"  Checkpoint version: {ckpt_version}")
    if '-dirty' in ckpt_version:
        print(f"Warning: [{tool_name}] Checkpoint was trained with uncommitted "
              f"changes ('-dirty'). Exact code state is not reproducible.")
    elif not is_release_version(ckpt_version):
        print(f"Warning: [{tool_name}] Checkpoint is not a release "
              f"(trained from an untagged commit with --experiment).")

    code_version = get_code_version()
    ckpt_major = parse_major(ckpt_version)
    code_major = parse_major(code_version)
    if ckpt_major is not None and code_major is not None and ckpt_major != code_major:
        print(f"Warning: [{tool_name}] MAJOR version mismatch: "
              f"checkpoint={ckpt_version}, code={code_version}. "
              f"state_dict may be incompatible.")
    return ckpt_version


def version_filename_part(version: Optional[str]) -> str:
    """Build the version suffix for artifact file names.

    e.g. 'v2.0.0' -> '_v2.0.0', None -> '' (checkpoint without an embedded version)
    """
    if not version:
        return ''
    # Replace characters that are not safe in file names (git describe output normally is)
    safe = re.sub(r'[^\w.\-]', '-', version)
    return f'_{safe}'
