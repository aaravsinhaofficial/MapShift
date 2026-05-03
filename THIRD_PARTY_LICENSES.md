# Third-Party Licenses

MapShift code is released under the MIT License in `LICENSE`. Generated benchmark outputs, raw episode records, health reports, rendered result tables, and rendered result figures produced by the artifact commands are released under CC BY 4.0 unless a downstream user applies a different license to their own generated derivative bundle.

This file records third-party dependencies and tooling used by the reviewed artifact path.

| Component | Role | License |
|---|---|---|
| Python | Runtime language | Python Software Foundation License |
| NumPy | Numerical arrays and statistics helpers | BSD-3-Clause |
| PyTorch | Learned-baseline implementation and tensor operations | BSD-style PyTorch license |
| setuptools | Build backend | MIT |
| Tectonic | Optional local paper PDF build tool | MIT |
| NeurIPS 2026 LaTeX style file | Paper formatting template | Distributed by NeurIPS for submission formatting |

The executable benchmark does not package a third-party dataset. Generated MapShift environments and tasks are produced from this repository's source code and configuration files.
