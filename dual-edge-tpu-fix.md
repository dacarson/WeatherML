# Dual Coral Edge TPU on Raspberry Pi 5 — Fix Documentation

## Hardware

| Component | Detail |
|---|---|
| SBC | Raspberry Pi 5 |
| HAT | Geekworm X1004 PCIe to Dual M.2 (ASMedia ASM1182e 2-port PCIe x1 Gen2 switch) |
| TPU | Coral M.2 Accelerator with Dual Edge TPU (E-key, via B+M adapter) |
| NVMe | NVMe SSD in second M.2 slot |
| PCIe uplink | Single PCIe 2.0 x1 (5 Gbps) from Pi 5 — shared by all downstream devices |
| Kernel | `6.12.75+rpt-rpi-v8` |

---

## Root Cause

The BCM2712 PCIe controller on the Pi 5 provides a single MSI domain with **32 slots**.

During boot, PCIe bridge ports (BCM2712 root port + ASM1182e switch ports) consumed **9 slots** via legacy MSI allocation — even though those interrupts are actually handled via GICv2 and the MSI slots go unused.

The original driver (`pcie-brcmstb.c`) used `bitmap_find_free_region()` which allocates power-of-2 aligned blocks:

- NVMe: separate domain (base `135790592`) — not relevant
- Bridges: 9 slots consumed (hwirq 0–8)
- TPU 0: needs 13 vectors → rounded up to 16-slot aligned block → allocated at hwirq 9–21 ✓
- TPU 1: needs 13 vectors → rounded up to 16-slot aligned block → only 10 slots remain → **`-ENOSPC`**

With the non-aligned `bitmap_find_next_zero_area()` patch, TPU 1 could fit into slots 22–31 (10 slots) — but still failed because **10 < 13**.

The complete fix required three changes working together.

---

## Initial State

Before the fix, the symptom was consistent:

- TPU 0 (`0001:06:00.0`) gets 13 MSI-X vectors and works at full speed (~2.58 ms / 387 FPS on MobileNet v2)
- TPU 1 (`0001:07:00.0`) gets 0 vectors and fails with `-ENOSPC` during driver init
- Both TPUs enumerate correctly in `lspci` and `/sys/class/apex/` shows both as **ALIVE** — but only with the correct M.2 slot assignment on the X1004 board: Dual Edge TPU in slot 1, NVMe SSD in slot 2. With the assignments reversed, PCIe enumeration only reached bus 4 and the second TPU was not visible
- `edgetpu_list_devices` returns 2 — all driver/sysfs state is healthy once enumeration is correct
- This is purely a Linux MSI vector allocation problem, not a hardware fault

The stock configuration had `pcie-msi-32` DT overlay active (4 GIC lines, domain = 32 slots) and `nvme.nr_io_queues=3` in `cmdline.txt` (ineffective — not honoured by the NVMe driver version in use). The `pcie-msi-32` overlay was later confirmed unnecessary — commenting it out had no effect on TPU initialization or slot count. The `pineboards-hat-ai` overlay or the base BCM2712 device tree already provides sufficient MSI interrupt lines for 32 slots on the Pi 5.

---

## What Was Tried and Ruled Out

| Approach | Outcome |
|---|---|
| Cap gasket driver to 8 vectors/TPU | Both TPUs init, but `libedgetpu` needs `TopLevelInterruptManager` (interrupts 8–11) — delegate returns null, devices unusable |
| `bypass_top_level=1` gasket param | Skips device reset/init — TPUs get vectors but library can't open devices |
| DT overlay: 4 GIC interrupt lines | Driver only uses index-1 line for MSI ISR regardless; domain still 32 slots. Overlay later confirmed unnecessary — `pineboards-hat-ai` or base BCM2712 DT already provides 32 slots on Pi 5 |
| `nvme.nr_io_queues=3` kernel param | NVMe still allocates 5 vectors — parameter not honoured by this driver version |
| `pci=nomsi` kernel param | Caused boot failure — NVMe or PCIe switch requires legacy MSI during initialization |
| Expanding MSI domain to 64 slots | Not possible — BCM2712 hardware uses a single 32-bit `MSI_INT_STATUS` register; domain is physically capped at 32 |
| Confirming NVMe/apex domain separation | NVMe MSI base `135790592` and apex MSI base `137363456` are confirmed separate domains — NVMe vectors do not consume apex slots |

---

## Changes Made

### 1. Bitmap allocator patch — `drivers/pci/controller/pcie-brcmstb.c`

Replace the power-of-2 aligned allocator with a contiguous-area allocator, and fix the failure return path:

```c
static int brcm_msi_alloc(struct brcm_msi *msi, unsigned int nr_irqs)
{
    int hwirq;

    mutex_lock(&msi->lock);

    hwirq = bitmap_find_next_zero_area(msi->used,
                                       msi->nr,
                                       0,
                                       nr_irqs,
                                       0);

    if (hwirq < msi->nr) {
        bitmap_set(msi->used, hwirq, nr_irqs);
        pr_info("brcm-msi: alloc %u vectors at hwirq=%d\n",
                nr_irqs, hwirq);
    } else {
        hwirq = -ENOSPC;    /* must return negative error */
    }

    mutex_unlock(&msi->lock);

    return hwirq;
}

static void brcm_msi_free(struct brcm_msi *msi, unsigned long hwirq,
                           unsigned int nr_irqs)
{
    mutex_lock(&msi->lock);
    bitmap_clear(msi->used, hwirq, nr_irqs);
    pr_info("brcm-msi: free %u vectors at hwirq=%lu\n", nr_irqs, hwirq);
    mutex_unlock(&msi->lock);
}
```

**Key detail:** `bitmap_find_next_zero_area()` returns `nbits` (not a negative number) on failure. The original patch used `if (hwirq >= 0)` which is always true and would call `bitmap_set()` out of bounds on failure. The correct check is `if (hwirq < msi->nr)`.

#### Build process

`CONFIG_PCIE_BRCMSTB` is `=y` (built-in) in the stock RPi kernel — a loadable module cannot override a built-in. The kernel must be rebuilt with `CONFIG_PCIE_BRCMSTB=m`:

```bash
# In /usr/src/linux-source-6.12
cp /boot/config-$(uname -r) .config
make olddefconfig
scripts/config --module CONFIG_PCIE_BRCMSTB
scripts/config --set-str CONFIG_LOCALVERSION "+rpt-rpi-v8"
make -j4 ARCH=arm64 Image.gz modules
make modules_install
cp arch/arm64/boot/Image.gz /boot/firmware/kernel8.img
```

The patched driver is then built and managed via DKMS:

```
/usr/src/pcie-brcmstb-patched-1.0/
    pcie-brcmstb.c      (patched source, with local pci.h copy)
    pci.h               (copied from drivers/pci/pci.h)
    Makefile
    dkms.conf
```

```bash
dkms add -m pcie-brcmstb-patched -v 1.0
dkms build -m pcie-brcmstb-patched -v 1.0 -k 6.12.75+rpt-rpi-v8
dkms install -m pcie-brcmstb-patched -v 1.0 -k 6.12.75+rpt-rpi-v8
```

The unpatched in-tree module must be removed so the initramfs doesn't load it first:

```bash
rm /lib/modules/6.12.75+rpt-rpi-v8/kernel/drivers/pci/controller/pcie-brcmstb.ko.xz
depmod -a 6.12.75+rpt-rpi-v8
update-initramfs -u -k 6.12.75+rpt-rpi-v8
```

---

### 2. ASMedia ASM1182e MSI quirk — `drivers/pci/quirks.c`

The ASM1182e PCIe switch has 5 ports (1 upstream + 4 downstream), each allocating 1–2 MSI slots during PCI enumeration. These slots are never actually used — PME is handled via GICv2. The quirk prevents this allocation entirely.

Add to the existing `quirk_no_msi` block (around line 1942):

```c
DECLARE_PCI_FIXUP_FINAL(PCI_VENDOR_ID_ASMEDIA, 0x1182, quirk_no_msi);
```

`PCI_VENDOR_ID_ASMEDIA` (`0x1b21`) is already defined in `include/linux/pci_ids.h`. No device ID constant exists for `0x1182` — use the raw hex.

This runs at `FIXUP_FINAL` time, after enumeration but before drivers claim devices.

After this change, rebuild the kernel image:

```bash
cd /usr/src/linux-source-6.12
touch drivers/pci/quirks.c
make -j4 ARCH=arm64 Image.gz
cp arch/arm64/boot/Image.gz /boot/firmware/kernel8.img
```

---

### 3. Kernel command line — `/boot/firmware/cmdline.txt`

Add `pci=noaer` to remove AER (Advanced Error Reporting) interrupt allocations from PCIe bridge ports:

```
console=serial0,115200 console=tty1 root=PARTUUID=30b7eac3-02 rootfstype=ext4 fsck.repair=yes rootwait cfg80211.ieee80211_regdom=US quiet splash plymouth.ignore-serial-consoles pci=noaer
```

**Note:** `pci=nomsi` was also tested but caused a boot failure — the NVMe or PCIe switch requires legacy MSI during initialization. `pci=noaer` alone (removing AER) is safe.

---

## Final MSI Slot Layout

After all three changes, the 32-slot domain is allocated as follows:

| Slots | Device | Count |
|---|---|---|
| 0–4 | PCIe bridge ports (BCM2712 root + residual) | 5 |
| 5–17 | TPU 0 (`apex0`, `0001:06:00.0`) | 13 |
| 18–30 | TPU 1 (`apex1`, `0001:07:00.0`) | 13 |
| 31 | Free | 1 |

---

## Verification

```bash
# Confirm both TPUs present
ls /sys/class/apex/
# Expected: apex_0  apex_1

# Confirm 26 apex MSI vectors allocated
cat /proc/interrupts | grep apex | wc -l
# Expected: 26

# Confirm patched allocator ran (no ENOSPC, no alignment waste)
dmesg | grep "brcm-msi"
# Expected: alloc lines starting at hwirq=5, then hwirq=18

# Confirm no initialization errors
dmesg | grep -E "apex|Couldn't initialize|ENOSPC"
```

---

## Boot Infrastructure Notes

The Raspberry Pi 5 firmware reads kernel and initramfs from `/boot/firmware/`, not `/boot/`:

| File | Purpose |
|---|---|
| `/boot/firmware/kernel8.img` | Kernel image (gzip compressed `Image.gz`) loaded for `rpi-v8` variant |
| `/boot/firmware/initramfs8` | initramfs — auto-synced from `/boot/initrd.img-6.12.75+rpt-rpi-v8` by `update-initramfs` |
| `/boot/firmware/config.txt` | Boot configuration (`kernel=kernel8.img`) |
| `/boot/firmware/cmdline.txt` | Kernel command line (single line, no trailing newline) |

The original kernel is backed up at `/boot/kernel8.img.orig`.

`auto_initramfs=1` in `config.txt` maps `kernel8.img` → `initramfs8` automatically by name convention.

---

## Maintenance

### Kernel package pinning (do this once)

Pin both the kernel image and headers to prevent `apt upgrade` from overwriting the custom `kernel8.img` or triggering DKMS rebuilds against a new kernel version:

```bash
sudo apt-mark hold linux-image-6.12.75+rpt-rpi-v8
sudo apt-mark hold linux-headers-6.12.75+rpt-rpi-v8
```

### Running `apt upgrade`

`apt upgrade` can destabilize the DKMS state even with the kernel pinned, because unrelated header packages trigger DKMS autoinstall against every kernel that has headers installed. The safest upgrade procedure is:

1. Run `apt upgrade` as normal — expect DKMS build failures for non-6.12 kernels in the output; these are harmless
2. After the upgrade completes, check DKMS state:
   ```bash
   dkms status
   ls /lib/modules/6.12.75+rpt-rpi-v8/updates/dkms/
   # Must show: apex.ko.xz  gasket.ko.xz  pcie-brcmstb.ko.xz
   ```
3. If `gasket` or `apex` are missing, rebuild manually (see gasket patches below)
4. Reboot and run the verification commands

**Watch out for:** apt removing `gasket-dkms` as part of a dependency chain (e.g. when removing old header packages). If this happens, reinstall with `sudo apt install gasket-dkms` and reapply the gasket patches below before rebuilding.

### Gasket source patches required for kernel 6.12+

The `gasket-dkms` 1.0-18 package from the Coral repo has three API incompatibilities with kernel 6.12. These patches must be applied to `/var/lib/dkms/gasket/1.0/source/` whenever gasket is reinstalled from scratch:

```bash
# 1. no_llseek removed in 6.12 — use noop_llseek
sudo sed -i 's/no_llseek/noop_llseek/g' \
    /var/lib/dkms/gasket/1.0/source/gasket_core.c

# 2. class_create() no longer takes a module owner argument (removed in 6.6+)
sudo sed -i 's/class_create(driver_desc->module, /class_create(/g' \
    /var/lib/dkms/gasket/1.0/source/gasket_core.c

# 3. eventfd_signal() no longer takes a count argument (removed in 6.10+)
sudo sed -i 's/eventfd_signal(ctx, 1)/eventfd_signal(ctx)/g' \
    /var/lib/dkms/gasket/1.0/source/gasket_interrupt.c
```

After patching, rebuild and install:

```bash
sudo dkms build gasket/1.0 -k 6.12.75+rpt-rpi-v8
sudo dkms install gasket/1.0 -k 6.12.75+rpt-rpi-v8
```

### pcie-brcmstb-patched and kernel image

The DKMS module (`pcie-brcmstb-patched 1.0`) will auto-rebuild on kernel updates via `dkms autoinstall`. However, the custom kernel image at `/boot/firmware/kernel8.img` will be overwritten by `apt upgrade` if the `linux-image` package is updated (pinning above prevents this). If a kernel upgrade is intentional:

- Rebuild `Image.gz` with the `quirks.c` patch and `CONFIG_PCIE_BRCMSTB=m` and redeploy
- The `quirks.c` change and `CONFIG_PCIE_BRCMSTB=m` config change live in `/usr/src/linux-source-6.12/` and must be reapplied if the source package is updated

### Cleaning up old kernel headers

Old header packages (`linux-headers-6.1.x-*`, `linux-kbuild-6.1`, etc.) accumulate over time and cause spurious DKMS build failures during `apt upgrade`. They are safe to remove with `apt autoremove`, but **watch the package list carefully** — apt may include `gasket-dkms` in the removal if `raspberrypi-kernel-headers` is also being removed (due to a declared dependency). If that happens, answer `n`, remove the header packages explicitly without `gasket-dkms`, then reinstall `gasket-dkms` and reapply the patches above.