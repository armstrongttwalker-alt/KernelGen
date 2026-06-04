# Supported hardware platforms

KernelGen internally integrates support for the following testing devices: Huawei Ascend, Hygon, Iluvatar, MetaX, Mthreads, Sunrise, and NVIDIA.

- **Generating Kernels**:
  - If users do not select a testing device, NVIDIA is used by default.
  - For generating FlagTree TLE operators specifically, the testing device can only be NVIDIA.
- **Optimizing Kernels**: Only supports Kernel optimization on NVIDIA.
- **Specializing Kernels**: Only supports Kernel specialization from NVIDIA to Huawei Ascend.