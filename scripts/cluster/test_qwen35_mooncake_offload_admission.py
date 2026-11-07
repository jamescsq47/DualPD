import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/compat/qwen35_mooncake/offload_admission.py"
)
SPEC = importlib.util.spec_from_file_location("offload_admission", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HybridOffloadAdmission = MODULE.HybridOffloadAdmission
HybridCapacityBudget = MODULE.HybridCapacityBudget
exact_mamba_snapshot_length = MODULE.exact_mamba_snapshot_length
stable_mamba_track_index = MODULE.stable_mamba_track_index
host_capacity_shortfall = MODULE.host_capacity_shortfall
restore_sources_available = MODULE.restore_sources_available
needs_internal_prefix_backup = MODULE.needs_internal_prefix_backup
is_structurally_attached = MODULE.is_structurally_attached


def test_mamba_snapshot_is_never_labeled_as_an_older_prefix():
    assert exact_mamba_snapshot_length(0, 64) == 0
    assert exact_mamba_snapshot_length(63, 64) == 0
    assert exact_mamba_snapshot_length(64, 64) == 64
    # A delayed snapshot at token 74 contains state through token 74.  It must
    # wait for token 128 instead of being mislabeled as the token-64 state.
    assert exact_mamba_snapshot_length(74, 64) == 0
    assert exact_mamba_snapshot_length(128, 64) == 128


def test_stable_mamba_track_index_selects_completed_slot():
    assert stable_mamba_track_index(0, 2) == 1
    assert stable_mamba_track_index(1, 2) == 0
    assert stable_mamba_track_index(0, 1) == 0


def test_one_snapshot_per_request_until_backup_finishes():
    gate = HybridOffloadAdmission()
    assert gate.can_submit("r1")
    gate.mark_submitted("r1")
    assert not gate.can_submit("r1")
    assert gate.can_submit("r2")
    gate.mark_backup_finished("r1")
    assert gate.can_submit("r1")


def test_capacity_failure_rate_limits_logging_without_changing_admission():
    gate = HybridOffloadAdmission(log_interval_s=5.0)
    assert gate.should_log_capacity_failure("r1", now=10.0)
    assert gate.can_submit("r1")
    assert not gate.should_log_capacity_failure("r1", now=10.25)
    assert gate.should_log_capacity_failure("r1", now=15.0)


def test_forget_clears_finished_request_state():
    gate = HybridOffloadAdmission()
    gate.mark_submitted("r1")
    gate.forget("r1")
    assert not gate.is_busy("r1")
    assert gate.can_submit("r1")


def test_global_capacity_failure_logging_is_shared_across_requests():
    gate = HybridOffloadAdmission(log_interval_s=5.0)
    assert gate.should_log_global_capacity_failure(now=10.0)
    assert not gate.should_log_global_capacity_failure(now=10.1)
    assert gate.should_log_global_capacity_failure(now=15.0)


def test_capacity_budget_is_deterministic_and_never_overcommits():
    budget = HybridCapacityBudget()
    budget.reset(kv_slots=192, mamba_slots=2)
    assert budget.reserve(64)
    assert budget.reserve(128)
    assert not budget.reserve(1)
    assert budget.kv_slots == 0
    assert budget.mamba_slots == 0


def test_only_unbacked_write_through_internal_prefix_needs_preservation():
    assert needs_internal_prefix_backup(
        backuped=False, child_count=1, write_policy="write_through"
    )
    assert not needs_internal_prefix_backup(
        backuped=True, child_count=1, write_policy="write_through"
    )
    assert not needs_internal_prefix_backup(
        backuped=False, child_count=0, write_policy="write_through"
    )
    assert not needs_internal_prefix_backup(
        backuped=False, child_count=1, write_policy="write_back"
    )


def test_host_capacity_shortfall_reclaims_only_missing_slots():
    assert host_capacity_shortfall(available=8192, needed=8192) == 0
    assert host_capacity_shortfall(available=6528, needed=8192) == 1664
    assert host_capacity_shortfall(available=10000, needed=8192) == 0


def test_restore_source_validation_turns_retired_hit_into_cache_miss():
    assert restore_sources_available(
        kv_sources_present=[True, True],
        mamba_required=True,
        mamba_present=True,
    )
    assert not restore_sources_available(
        kv_sources_present=[True, False],
        mamba_required=False,
        mamba_present=False,
    )
    assert not restore_sources_available(
        kv_sources_present=[],
        mamba_required=True,
        mamba_present=False,
    )


def test_structural_attachment_uses_object_identity_not_just_child_key():
    class Node:
        def __init__(self, key=None, parent=None):
            self.key = key
            self.parent = parent
            self.children = {}

    parent = Node()
    current = Node("same-key", parent)
    retired = Node("same-key", parent)
    parent.children["same-key"] = current

    assert is_structurally_attached(current, lambda key: key)
    assert not is_structurally_attached(retired, lambda key: key)
    assert not is_structurally_attached(parent, lambda key: key)
