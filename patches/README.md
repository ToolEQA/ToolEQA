# Local runtime dependency patches

These text patches preserve the runtime source changes used by ToolEQA without
publishing third-party caches, weights or local installation artifacts. Submodule
pointers are unchanged. Initialize the submodules first, then check and apply:

```bash
git -C third_party/verl apply --check ../../patches/verl-runtime.patch
git -C third_party/verl apply ../../patches/verl-runtime.patch
git -C third_party/DetAny3D apply --check ../../patches/detany-runtime.patch
git -C third_party/DetAny3D apply ../../patches/detany-runtime.patch
git -C third_party/DetAny3D/UniDepth apply --check ../../../patches/unidepth-dependencies.patch
git -C third_party/DetAny3D/UniDepth apply ../../../patches/unidepth-dependencies.patch
```

Base revisions:

- verl: `0ae27589903cae94006cb282f3925a9885482195`
- DetAny3D: `5299c83d2999a4b95431c31e28d065cdfe44c07e`
- UniDepth: `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`

Do not reapply to this working copy: its dependencies already contain these
changes. The patches intentionally omit bytecode, downloaded model weights and
GroundingDINO installation/deletion state. Training launchers still contain
machine-specific data/model paths, GPU UUIDs and runtime-library locations; adapt
these for another machine. No datasets, checkpoints or generated libraries are
included.
