# Board validation — 2026-07-21

Target: `phytiumpi`, Linux `6.6.63-phytium-embedded-v3.3`, AArch64.

## CPU1 firmware baseline

- Deployed firmware: `/lib/firmware/openamp_core0.elf`
- SHA-256: `39c335e7a9c560d40fa4522e5ef0173ba2326caf5fcae93b5df6e5feecc4aadb`
- Protocol markers extracted from the deployed ELF:
  - `EMG_PROTOCOL_MAGIC = 0x31474D45`
  - `EMG_PROTOCOL_VERSION = 1`
  - `EMG_ACK_CONFIG_FLAGS = 0x43464731`
- DWARF structure sizes extracted from the deployed ELF:
  - `EmgPacketHeader`: 32 bytes
  - `EmgConfigPacket`: 48 bytes
  - `EmgRawChunkPacket`: 164 bytes
  - `EmgChannelFeatureWire`: 20 bytes
  - `EmgFeaturePacket`: 116 bytes
  - `EmgHeartbeatPacket`: 40 bytes
  - `EmgErrorPacket`: 84 bytes
- Header offsets: magic 0, version 4, header_size 6, msg_type 8,
  total_size 10, seq 12, host_ts_ns 16, sample_rate_hz 24,
  channel_count 28, reserved 30.

## RPMsg v1 result

- `remoteproc0`: running; firmware channel `rpmsg-openamp-demo-channel`.
- Three consecutive cycles passed CONFIG ACK, RAW send and FEATURE receive.
- Each cycle produced one valid FEATURE frame from 200 raw samples.
- Invalid packet count: 0; CPU1 protocol error count: 0.
- A real endpoint leak was found during the first run. The client now issues
  `RPMSG_DESTROY_EPT_IOCTL` before closing the endpoint descriptor.
- After the fix, repeated cycles leave only `/dev/rpmsg_ctrl0`; no dynamic
  `/dev/rpmsgN` endpoint remains.

## Report layout result

The 7.9 MB real offline action report containing 17 images was rendered through
the board's Qt report page:

| Window | Report viewport limit | Maximum image width | Result |
|---|---:|---:|---|
| 800x600 | 371 px | 371 px | pass |
| 1280x720 | 851 px | 851 px | pass |
| 2180x1271 | 1751 px | 1751 px | pass |

Evidence is stored on the board in
`/home/user/stroke-rehab-runtime/report-layout-evidence/`.

## Remaining hardware blocker

Repeated real RGB-D capture cannot currently complete because the Astra Pro
USB link enters a control-transfer timeout, both before and after physical
re-enumeration:

```text
uvcvideo: Failed to set UVC probe control: -110 (exp. 26)
hub: hub_ext_port_status failed (err = -110)
```

The protected validation process was terminated and no application process was
left holding the camera. Multi-cycle capture validation remains pending until
the USB power/hub/controller fault is resolved.
