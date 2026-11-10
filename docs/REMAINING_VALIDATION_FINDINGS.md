# Findings and subsequent verification

Current status: stages1–6 and stage7a are independently accepted on the local
TP1 BrowseComp comparison; stage7a Decode11193.29 token/s,−0.64% versus ORIGINAL,
100% reuse and clean final gates. Stage7b historical archival is independently
accepted at11279.34 Decode token/s (+0.13% versus ORIGINAL),100% reuse and clean
final gates under full300+1200. The later installed-Router entry comparison
passes at11200.41 Decode token/s (−0.57% versus ORIGINAL),100% reuse and clean
gates. Dense Qwen3-8B same-hostTCP/TP2 Direct and D2P Host two-turn functionality
is independently verified by `dense-tp2-native-r7-20261110`, using the existing
per-node single-NIC configuration knob; see FINAL_ENGINE_VALIDATION_PLAN.md.
Hybrid/Mamba/MoE, TCP P2D Host spill and final package/source integration remain
open; historical operational suffix execution
is not established by the archival source/contract tests.
Stage6 formally fixes trusted-profile-loader
partial application and three same-IP listener checks, and extracts16 common
case arguments. CPU928/native1018/CUDA2, whole-source proofs and independent
CODE GO pass; the new full300+1200 comparison is accepted at11212.44 Decode
token/s (−0.47% versus ORIGINAL),100% parent reuse and clean final gates.
Historical observations
below are retained as chronology, not a claim that those fixes are unapplied.
The findings below retain their original chronology. They do not override the
dated accepted stages or claim the whole goal is complete.

## Historical read-only findings during scheduler-workset-r1

No runtime edits in this note. The active model case freezes engine and frontend
serving sources until its final result and cleanup acceptance.

## Historical multi-node smoke is not current TCP acceptance

scripts/cluster/multinode_smoke.py directly reads cfg.control_root and creates
AgenticEarlyClaimStore / SharedHostStagingLedger under that shared directory.
It expects direct_complete/host_ready routes from local files. In contrast,
maintained multinode.py requires group_control.enabled and sets V2 environment
without any runtime ledger, marker or shared-control-root configuration.
Consequently, do not invoke that historical smoke and claim current TCP model
coverage. Stage7 must explicitly classify this entry and preserve historical
replay; representative V2 coverage needs the actual group protocol/controller
and final callbacks, with no fabricated local route files.

The current multinode launcher does allow distinct logical endpoint groups on
one physical host if GPU groups and all listener ports are nonoverlapping.
Thus the dedicated eight-H100 allocation can validate the real TCP path without
borrowing another node. TP>1/fault coverage, exact token comparison, Direct/Host
route evidence and process/port/helper cleanup require their own audited plan.
This is a feasibility finding, not actual runtime coverage or performance proof.

## Remaining Scheduler methods

After moving the lease/broker, a name-based exploratory grouping finds25
Direct-named methods (~2850 lines),10 TP-prefix methods (~1292 lines),17 other
agentic/drain methods (~1064 lines). This is NOT an extraction inventory:
mixed native methods and methods with agentic elsewhere in the name remain.
Derive complete ownership-based groups from the current AST/call graph before
editing, preserving eager annotations, decorators and explicit native hooks.

There remain seven explicit Scheduler attribute references: five method
references inside four methods, and two control constants in TP consume.
Namespace fixtures and native-module private hook patches require an explicit
compatibility design; do not leave large agentic bodies inline merely because
their old globals are difficult, or silently make fixtures require a mixin
instance. Exact group/body/module reconstruction remains required.

## Configuration scope: active CLI versus historical shell defaults

Read-only inspection during scheduler-components-r2 finds
qwen35_slurm.py --fast-tool-seconds default1.0, whereas common historical shell
wrappers including run_4p4d_numa_case.sh still fall back to2 for both
FAST_TOOL_THRESHOLD_SECONDS and DIRECT_WAIT_SECONDS. Several archived model
recipes also explicitly set2; the active cleanup comparison explicitly sets1/1
and is unaffected. These are entry-specific defaults, not evidence that any
accepted comparison silently used2 or30 seconds.

Stage6/7 must distinguish maintained current defaults from historical replay
profiles, preserve explicit overrides and label the selected scope. Do not
blindly merge all parsers/defaults, rewrite recorded experiments, or silently
change a recipe while claiming source/config identity. The current serving
source remains frozen and no default is changed by this finding.

## Profile loader: reproducible partial-application failure

Read-only characterization of the actual maintained
scripts/new_method/internal/apply_experiment_profile.sh reproduces two failures:
an invalid later scalar (TEMPERATURE0 followed by TOP_Pnull) makes the Python
loader raise after one output row, but the Bash process-substitution loop still
applies TEMPERATURE, selects EXPERIMENT_CONFIG and returns0. Even an unknown key
rejected before output still selects the profile and returns0. Callers using
set-e therefore cannot reliably stop an invalid launch. This is a configuration
boundary defect, not evidence of a fault in the active fully valid explicit plan.

External configuration-draft/test_profile_transaction.py records both current
behaviors. An UNAPPLIED candidate captures the complete loader output and checks
its exit status before mutating any configured environment or selecting the
profile. Five candidate contract groups cover success/path normalization,
explicit overrides, six invalid shapes, strict-caller stopping and missing loader.
They pass after correcting one new fixture assumption: PROFILE_ORIGIN labels
are intentionally shell-local, not exported; tests inspect the real shell value
without introducing a new export. No serving file or default was changed.
Candidate path: pool004 runs/engine-cleanup-validation-20261108/configuration-draft.
Independent review, formal application, retained tests and full gates remain.

The external candidate is now independently rechecked across10 groups: valid
inputs compared directly with the frozen old function (default schema, explicit
empty overrides, empty strings and literal shell-special text), relative symlink
selection, missing parent/loader/python, preserved pre-existing origin, strict
and conditional failure calls, and same-shell success followed by invalid input.
All pass; no eval/new export/default change is introduced. The guarantee is
limited to realpath/trusted-loader failures before applying settings, not a
general rollback for later shell assignment/export failures or an arbitrary
replacement loader falsely returning success with malformed records.

The active frozen BrowseComp plan has EXPERIMENT_CONFIG unset and99 explicitly
recorded environment entries. It does not exercise this optional profile loader;
do not attribute its throughput result to profile-parser coverage. Formal stage6
must retain the specific shell contract tests and document whether a profile-
driven model launch is additionally exercised. Earlier recipe/default conflicts
remain a separate consolidation task; this focused fix alone is not stage6 done.

## Shared case configuration candidate, still unapplied

The four maintained case wrappers (1P3D,1P5D,2P6D,4P4D) repeat16 exactly equal
argv assignments. A separate common-case-candidate extracts only those16 into
one Bash array builder, evaluated after any existing profile application. Every
member is a literal or side-effect-free source-knob/default expansion; no eval,
export, policy owner or new worker is introduced. GPU/port/topology/per-case
capacity and local/TCP policy differences remain in their own existing entries.
Historical2-second fallbacks remain unchanged in this extraction; current formal
profiles/comparisons explicitly set1/1. Later maintained/replay classification
must describe this scope rather than silently rewrite historical settings.

The candidate also contains the already reviewed profile failure fix and an
unchanged loader/schema/allowlist. An immutable0ed608d whole-file proof checks
all four complete wrappers, the whole helper/profile function and the loader.
Two structural test groups reject11 destructive mutations. Eight actual Bash
capture groups compare29 original/candidate launches: all explicit bindings and
pipeline argv match after path-only relocation, with no duplicate target keys.
Cases cover defaults, explicit/empty/shared literal-special knobs, raw target
precedence and profile ordering/overrides. They preserve the existing fact that
only4P4D applies this profile at the case layer; the other three are not silently
given new profile semantics. All87 maintained Shell files also passbash-n.

Capture tests deliberately stub fingerprint checks and intercept env before the
pipeline: they prove Bash argument construction, not environment correctness,
native imports, model behavior or GPU acceptance. Current frozen old files are
checked against immutable Git before comparison. Formal retained tests must copy
their old side from immutable history (including the old profile function), use
automatic checkout discovery and include the new helper in source/package/replay
inventories. Comparing an applied new function with itself is not valid coverage.
The current model remains frozen on stage5; no configuration candidate is applied.

A further unapplied validation candidate changes only three expressions in
multinode.load_config: Router and relay worker-port sets cover every group with
the same explicit host_ip, and relay/Router co-location compareshost_ip rather
than logicalnode_id. Ten real-parser groups pass, including17 old-accepted
cross-group collisions now rejected, third-group/swapped-owner cases, preserved
different-host reuse and unchanged previous rejection messages. Nine valid
model/TP parsed configurations and complete generated plans remain identical.
The original nine groups and complete-module three-edit proof have independent
PREP GO; the added symmetric group is a further main-side check. This stays
configuration validation only: no DNS/multi-NIC/physical-host identity discovery,
socket probe, launch or GPU test. Same-host actual TCP validation must consistently
use one explicit routable IPv4; different IP aliases cannot be auto-merged.
