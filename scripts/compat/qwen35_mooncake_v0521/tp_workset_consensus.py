"""Pure helpers for the v0.5.21 P->D TP workset compatibility fix."""

from __future__ import annotations

import torch


def resolve_transfer_workset(
    gathered: torch.Tensor,
    local_entries: list[tuple[int, int, int]],
    *,
    waiting_status: int,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Resolve fixed-slot TP poll rows into local queue order.

    ``gathered`` has shape ``[tp, metadata_slots, 3]``, with columns
    present/signature/poll. A room may advance only when every TP rank owns
    the same room in the same metadata slot.
    """
    resolved = []
    mismatches = []
    for metadata_index, signature, local_poll in local_entries:
        rows = gathered[:, metadata_index, :]
        present = rows[:, 0]
        signatures = rows[:, 1]
        if bool(torch.all(present == 1).item()) and bool(
            torch.all(signatures == signature).item()
        ):
            resolved.append(int(torch.min(rows[:, 2]).item()))
        else:
            # Missing/different work on a peer is admission skew, not transfer
            # failure. Retain every resource and wait for the peer.
            resolved.append(int(waiting_status))
            mismatches.append((metadata_index, signature, int(local_poll)))
    return resolved, mismatches


def resolve_identity_workset(
    gathered: torch.Tensor,
    local_entries: list[tuple[int, int]],
    *,
    waiting_status: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Resolve an unordered, fixed-capacity TP workset by request identity.

    ``gathered`` has shape ``[tp, capacity, 2]`` with signature/status rows;
    a zero signature is an unused row.  Queue positions may differ between TP
    ranks.  A local request advances only after its signature occurs exactly
    once on every rank, at which point the minimum status is authoritative.
    """
    rank_maps: list[dict[int, int]] = []
    duplicate_signatures: list[set[int]] = []
    for rank_rows in gathered:
        statuses: dict[int, int] = {}
        duplicates: set[int] = set()
        for signature, status in rank_rows.tolist():
            signature = int(signature)
            if signature == 0:
                continue
            if signature in statuses:
                duplicates.add(signature)
            else:
                statuses[signature] = int(status)
        rank_maps.append(statuses)
        duplicate_signatures.append(duplicates)

    resolved: list[int] = []
    mismatches: list[tuple[int, int]] = []
    for signature, local_status in local_entries:
        if any(
            signature not in statuses or signature in duplicates
            for statuses, duplicates in zip(rank_maps, duplicate_signatures)
        ):
            resolved.append(int(waiting_status))
            mismatches.append((signature, int(local_status)))
            continue
        resolved.append(min(statuses[signature] for statuses in rank_maps))
    return resolved, mismatches
