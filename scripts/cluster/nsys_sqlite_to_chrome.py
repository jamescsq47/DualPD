#!/usr/bin/env python3
"""Convert one or more Nsight Systems SQLite exports to Chrome Trace JSON.

The JSON produced by ``nsys export --type=json`` is protobuf JSON, not the
Chrome Trace Event format accepted by edge://tracing.  This converter keeps
the useful CUDA/NVTX timeline records and writes a streaming trace so large
100-second captures do not have to fit in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator


def tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def strings(connection: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in tables(connection):
        return {}
    return {
        int(identifier): str(value)
        for identifier, value in connection.execute("SELECT id, value FROM StringIds")
    }


def choose(existing: set[str], *candidates: str) -> str | None:
    return next((candidate for candidate in candidates if candidate in existing), None)


def timed_rows(
    connection: sqlite3.Connection,
    table: str,
    fields: Iterable[str],
) -> Iterator[dict[str, Any]]:
    existing = columns(connection, table)
    requested = [field for field in fields if field in existing]
    if "start" not in requested or "end" not in requested:
        return
    query = ", ".join(f'"{field}"' for field in requested)
    for row in connection.execute(f'SELECT {query} FROM "{table}" ORDER BY start'):
        yield dict(zip(requested, row, strict=True))


def string_value(mapping: dict[int, str], value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    try:
        return mapping.get(int(value), fallback)
    except (TypeError, ValueError):
        return str(value)


def complete_event(
    name: str,
    category: str,
    start: int,
    end: int,
    process: str,
    thread: str,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": start / 1000.0,
        "dur": max(0, end - start) / 1000.0,
        "pid": process,
        "tid": thread,
        "args": args or {},
    }


def session_epoch_ns(database: Path) -> int:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    if "TARGET_INFO_SESSION_START_TIME" not in tables(connection):
        connection.close()
        return 0
    row = connection.execute(
        "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1"
    ).fetchone()
    connection.close()
    return int(row[0]) if row and row[0] is not None else 0


def events(database: Path, label: str, time_offset_ns: int) -> Iterator[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    available = tables(connection)
    names = strings(connection)

    yield {
        "name": "process_name",
        "ph": "M",
        "pid": label,
        "tid": 0,
        "args": {"name": label},
    }

    kernel_tables = sorted(
        table for table in available if table.startswith("CUPTI_ACTIVITY_KIND_KERNEL")
    )
    for table in kernel_tables:
        existing = columns(connection, table)
        name_column = choose(existing, "demangledName", "shortName", "nameId", "name")
        fields = [
            "start", "end", "deviceId", "contextId", "streamId", "correlationId",
            "globalPid", *(filter(None, [name_column])),
        ]
        for row in timed_rows(connection, table, fields):
            raw_name = row.get(name_column) if name_column else None
            kernel_name = string_value(names, raw_name, "CUDA kernel")
            device = row.get("deviceId", "?")
            stream = row.get("streamId", "?")
            yield complete_event(
                kernel_name,
                "CUDA Kernel",
                time_offset_ns + int(row["start"]),
                time_offset_ns + int(row["end"]),
                label,
                f"GPU {device} / stream {stream}",
                {key: value for key, value in row.items() if key not in {"start", "end", name_column}},
            )

    for table, category in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUDA Memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "CUDA Memset"),
    ):
        if table not in available:
            continue
        fields = [
            "start", "end", "deviceId", "contextId", "streamId", "correlationId",
            "copyKind", "srcKind", "dstKind", "bytes", "value",
        ]
        for row in timed_rows(connection, table, fields):
            device = row.get("deviceId", "?")
            stream = row.get("streamId", "?")
            kind = row.get("copyKind")
            name = category if kind is None else f"{category} kind={kind}"
            yield complete_event(
                name,
                category,
                time_offset_ns + int(row["start"]),
                time_offset_ns + int(row["end"]),
                label,
                f"GPU {device} / stream {stream}",
                {key: value for key, value in row.items() if key not in {"start", "end"}},
            )

    if "CUPTI_ACTIVITY_KIND_RUNTIME" in available:
        table = "CUPTI_ACTIVITY_KIND_RUNTIME"
        existing = columns(connection, table)
        name_column = choose(existing, "nameId", "name")
        fields = [
            "start", "end", "globalTid", "correlationId", "returnValue",
            *(filter(None, [name_column])),
        ]
        for row in timed_rows(connection, table, fields):
            raw_name = row.get(name_column) if name_column else None
            yield complete_event(
                string_value(names, raw_name, "CUDA API"),
                "CUDA API",
                time_offset_ns + int(row["start"]),
                time_offset_ns + int(row["end"]),
                label,
                f"CPU thread {row.get('globalTid', '?')}",
                {key: value for key, value in row.items() if key not in {"start", "end", name_column}},
            )

    if "NVTX_EVENTS" in available:
        table = "NVTX_EVENTS"
        existing = columns(connection, table)
        text_column = choose(existing, "text", "textId", "nameId")
        fields = [
            "start", "end", "globalTid", "globalPid", "eventType", "domainId",
            *(filter(None, [text_column])),
        ]
        for row in timed_rows(connection, table, fields):
            if row.get("end") is None:
                continue
            raw_name = row.get(text_column) if text_column else None
            if text_column == "text":
                event_name = str(raw_name or "NVTX")
            else:
                event_name = string_value(names, raw_name, "NVTX")
            yield complete_event(
                event_name,
                "NVTX",
                time_offset_ns + int(row["start"]),
                time_offset_ns + int(row["end"]),
                label,
                f"CPU thread {row.get('globalTid', '?')}",
                {key: value for key, value in row.items() if key not in {"start", "end", text_column}},
            )

    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("databases", nargs="+", type=Path)
    args = parser.parse_args()
    missing = [str(path) for path in args.databases if not path.is_file()]
    if missing:
        raise SystemExit("missing SQLite export(s): " + ", ".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    epochs = {database: session_epoch_ns(database) for database in args.databases}
    nonzero_epochs = [epoch for epoch in epochs.values() if epoch]
    origin_epoch_ns = min(nonzero_epochs) if nonzero_epochs else 0
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        output.write('{"displayTimeUnit":"ms","traceEvents":[')
        first = True
        for database in args.databases:
            label = database.stem
            offset = epochs[database] - origin_epoch_ns if epochs[database] else 0
            for event in events(database, label, offset):
                if not first:
                    output.write(",")
                json.dump(event, output, ensure_ascii=False, separators=(",", ":"))
                first = False
                count += 1
        output.write("]}\n")
    if count <= len(args.databases):
        raise SystemExit("no CUDA/NVTX events were exported")
    print(json.dumps({"output": str(args.output), "events": count}))


if __name__ == "__main__":
    main()
