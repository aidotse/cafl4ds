# Developing

## On the Edgelab's Gaudi VM

This repo provides a Docker development environment for Gaudi HPUs: `docker/gaudi.env.Dockerfile`.

### 1. Build the image

Follow the header comment in `docker/gaudi.env.Dockerfile`. The base image tag **must match the host driver version**
(check with `hl-smi`). By default the the built image name is `gaudi-env-cafl4ds:latest`.

### 2. Check the hardware

```bash
HABANA_LOGS=/tmp/hllog hl-smi          # no sudo needed; the log dir just has to be writable
```

Should list 2× `HL-225` (Gaudi 2) cards. `hl-smi -q -d NET` shows NIC/MAC info; `hl-smi -n link -i <pci-addr>` shows
port link state.

### 3. Run something on an HPU

`scripts/run_gaudi_dev.sh <image> <device_id|all> [command...]` wraps `docker run` with the flags Gaudi needs
(`--runtime=habana`, memory pinning, RoCE interconnect, recipe-cache isolation, etc. — all documented inline in the
script). It auto-detects whether stdin is a TTY, so it works both interactively and headlessly.

```bash
# Single-card smoke test (loads PyTorch, allocates a tensor on hpu:0)
./scripts/run_gaudi_dev.sh gaudi-env-cafl4ds:latest 0 python scripts/gaudi_simple_test.py

# Interactive shell on card 1
./scripts/run_gaudi_dev.sh gaudi-env-cafl4ds:latest 1 bash

# Use the whole node (both cards + RoCE fabric)
./scripts/run_gaudi_dev.sh gaudi-env-cafl4ds:latest all python -m your.multi_hpu.entrypoint
```

The project root is mounted at `/workspace` (the container's workdir), so pass paths relative to the repo root.

### Isolation: single-card vs. `all`

- **`<device_id>` (e.g. `0`)** exposes *only* `/dev/accel/accel<id>`, so the container sees exactly one card. HCL never
    tries to bring up the inter-card fabric — this is the simplest, most robust mode for single-HPU work.
- **`all`** exposes the whole `/dev/accel` dir plus `/dev/infiniband` for the RoCE interconnect — use this for
    multi-card / DDP runs.

> ⚠️ Don't mix the two: exposing both cards while pinning `HABANA_VISIBLE_DEVICES` to a single id gives HCL an
> inconsistent view and it segfaults during device acquire (`getMacInfo → readMacInfoFromFile failed`). The script
> already keeps these consistent — just don't hand-roll the `docker run`.

### Host prerequisites (RDMA)

Device acquire brings up Gaudi's RoCE/ibverbs stack even for a single card, so the host needs the RDMA user-access layer
present:

- `ls /dev/infiniband/` shows `uverbs0`, `uverbs1`, …
- `ls /sys/class/infiniband/` shows `hbl_0`, `hbl_1`
- `lsmod | grep -E 'ib_uverbs|habanalabs_ib'` are loaded

If these are missing (`Device acquire failed`, `ibv initialization failed`), the Habana driver stack needs
(re)installing on the host — that's a root-level host fix, not a container-flag issue.
