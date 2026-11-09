# Read-only findings during scheduler-workset-r1

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
