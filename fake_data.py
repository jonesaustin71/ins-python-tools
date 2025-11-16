#!/usr/bin/env python3
"""Synthetic GNSS/IMU/PPS record generator for Teensy INS logs."""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

MAGIC = 0x54494E53
LOG_TYPE_GNSS_IMU = 0x0101
FILE_MAGIC = 0x544C4F47

HEADER_STRUCT = struct.Struct('<I H H Q I')
FILE_HEADER_STRUCT = struct.Struct('<I H H Q')
BOOL_STRUCT = struct.Struct('<?')
DOUBLE_STRUCT = struct.Struct('<d')
FLOAT_STRUCT = struct.Struct('<f')


def _align_length(buffer: bytearray, alignment: int) -> None:
    padding = (alignment - (len(buffer) % alignment)) % alignment
    if padding:
        buffer.extend(b"\x00" * padding)


def _pack_field(buffer: bytearray, fmt: str, value, alignment: int) -> None:
    _align_length(buffer, alignment)
    buffer.extend(struct.pack(fmt, value))


@dataclass
class GnssSample:
    iTOW_ms: int
    tow_s: float
    lat_deg: float
    lon_deg: float
    height_m: float
    hmsl_m: float
    velN_mps: float
    velE_mps: float
    velD_mps: float
    gSpeed_mps: float
    headMot_deg: float
    headVeh_deg: float
    hAcc_m: float
    vAcc_m: float
    sAcc_mps: float
    headAcc_deg: float
    numSV: int
    fixType: int
    flags: int
    flags2: int
    pDOP: float
    capture_time_us: int
    pps_time_us: int
    pps_offset_us: int
    pps_count_snapshot: int
    valid: bool

    def pack(self) -> bytes:
        buf = bytearray()
        _pack_field(buf, "<I", self.iTOW_ms, 4)
        _pack_field(buf, "<d", self.tow_s, 8)
        for value in (
            self.lat_deg,
            self.lon_deg,
            self.height_m,
            self.hmsl_m,
            self.velN_mps,
            self.velE_mps,
            self.velD_mps,
            self.gSpeed_mps,
            self.headMot_deg,
            self.headVeh_deg,
            self.hAcc_m,
            self.vAcc_m,
            self.sAcc_mps,
            self.headAcc_deg,
        ):
            _pack_field(buf, "<d", value, 8)
        _pack_field(buf, "<B", self.numSV, 1)
        _pack_field(buf, "<B", self.fixType, 1)
        _pack_field(buf, "<B", self.flags, 1)
        _pack_field(buf, "<B", self.flags2, 1)
        _pack_field(buf, "<H", int(self.pDOP * 100), 2)
        _pack_field(buf, "<Q", self.capture_time_us, 8)
        _pack_field(buf, "<Q", self.pps_time_us, 8)
        _pack_field(buf, "<i", self.pps_offset_us, 4)
        _pack_field(buf, "<I", self.pps_count_snapshot, 4)
        _pack_field(buf, "<?", self.valid, 1)
        _align_length(buf, 8)
        return bytes(buf)


def pack_imu_sample(timestamp_us: int) -> bytes:
    base_rate = 0.001 * math.sin(timestamp_us / 1e6)
    payload = bytearray()
    _pack_field(payload, "<Q", timestamp_us, 8)
    _pack_field(payload, "<I", 0x174, 4)
    for axis in range(3):
        rate = base_rate + axis * 0.01
        accel = 0.01 * math.cos(timestamp_us / 1e6 + axis)
        _pack_field(payload, "<f", rate, 4)
        _pack_field(payload, "<f", accel, 4)
    _pack_field(payload, "<?", True, 1)
    _align_length(payload, 8)
    return bytes(payload)


def make_record(seq: int, timestamp_us: int, sample: GnssSample) -> Tuple[bytes, bytes, bytes]:
    payload = bytearray()
    payload.extend(struct.pack('<Q', timestamp_us))
    payload.extend(sample.pack())
    payload.extend(pack_imu_sample(timestamp_us))
    payload_bytes = bytes(payload)
    header = HEADER_STRUCT.pack(MAGIC, LOG_TYPE_GNSS_IMU, len(payload_bytes), timestamp_us, seq)
    crc = zlib.crc32(payload_bytes, zlib.crc32(header)) & 0xFFFFFFFF
    crc_bytes = struct.pack('<I', crc)
    return header, payload_bytes, crc_bytes


def synthesize_records(count: int, start_tow: float, rate_hz: float) -> Iterable[Tuple[bytes, bytes]]:
    timestamp = 0
    tow = start_tow
    dt = 1.0 / rate_hz
    for idx in range(count):
        lat = 40.735 + 0.0001 * math.sin(0.1 * idx)
        lon = -89.61 + 0.0001 * math.cos(0.1 * idx)
        height = 240 + 0.5 * math.sin(0.05 * idx)
        velN = 5 + math.sin(0.05 * idx)
        velE = 0.5 * math.cos(0.05 * idx)
        speed = math.hypot(velN, velE)
        head = math.degrees(math.atan2(velE, velN)) % 360
        gnss = GnssSample(
            iTOW_ms=int(tow * 1000) % 604800000,
            tow_s=tow,
            lat_deg=lat,
            lon_deg=lon,
            height_m=height,
            hmsl_m=height - 30,
            velN_mps=velN,
            velE_mps=velE,
            velD_mps=0.0,
            gSpeed_mps=speed,
            headMot_deg=head,
            headVeh_deg=head,
            hAcc_m=0.02,
            vAcc_m=0.05,
            sAcc_mps=0.03,
            headAcc_deg=0.1,
            numSV=20,
            fixType=5,
            flags=0x31,
            flags2=0x07,
            pDOP=1.2,
            capture_time_us=timestamp,
            pps_time_us=(timestamp // 1000000) * 1000000,
            pps_offset_us=int(timestamp % 1000000) - 1000000,
            pps_count_snapshot=idx,
            valid=True,
        )
        header, payload, crc_bytes = make_record(idx, timestamp, gnss)
        yield header, payload, crc_bytes
        timestamp += int(dt * 1_000_000)
        tow += dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate synthetic Teensy INS logs')
    parser.add_argument('-n', '--count', type=int, default=100, help='Number of records to generate')
    parser.add_argument('--rate', type=float, default=10.0, help='GNSS/IMU rate in Hz')
    parser.add_argument('--start-tow', type=float, default=0.0, help='Starting GPS time-of-week in seconds')
    default_path = Path(__file__).with_name("data").joinpath("synthetic_log.bin")
    parser.add_argument('-o', '--output', type=Path, default=default_path, help='Output binary log path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        file_header = FILE_HEADER_STRUCT.pack(FILE_MAGIC, 1, FILE_HEADER_STRUCT.size, 0)
        f.write(file_header)
        for header, payload, crc_bytes in synthesize_records(args.count, args.start_tow, args.rate):
            f.write(header)
            f.write(payload)
            f.write(crc_bytes)
    print(f'Wrote {args.count} synthetic records to {args.output}')


if __name__ == '__main__':
    main()
