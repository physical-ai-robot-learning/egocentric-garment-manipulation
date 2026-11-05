# upstream/ — third-party licences for code conventions this repository follows

| folder | upstream | licence | what in this repository relates to it |
|---|---|---|---|
| `umi/` | Universal Manipulation Interface, <https://github.com/real-stanford/universal_manipulation_interface> (`umi/common/pose_util.py`, last upstream change `c34ba9c`, 2024-02-16) | MIT ([`umi/LICENSE`](umi/LICENSE), copied from upstream) | the rot6d convention of the action contract (`src/laundry_shared/contracts/action_contract.py`: first two rows of R). Until 2026-11-05 also `src/umi/common/pose_util.py`, our own minimal stand-in with the same function names and conventions (`pose_to_mat`, `mat_to_rot6d`, `rot6d_to_mat`, `mat_to_pose10d`, `pose10d_to_mat`; not a copy of upstream code), removed from the tree because the pipeline does not import it |

The HandUMI hardware licence (Apache-2.0) stays with its build copy in [`hardware/umihand/`](../hardware/umihand/README.md).
Provenance (source path + md5) of every vendored module: [`docs/provenance.md`](../docs/provenance.md).
