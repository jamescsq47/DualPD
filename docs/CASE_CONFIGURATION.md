# Shared case configuration and validation boundaries

Four local case launchers now build16 identical environment argv entries through
scripts/new_method/internal/agentic_case_environment.sh. It constructs a Bash
array only: no export, eval, process, model initialization or ownership state.
Quoted array expansion retains oneKEY=VALUE argument even for spaces or literal
shell metacharacters. Topology, GPUs, ports, Host capacity, admission/backpressure
and all other case-specific settings remain in their original entries.

Existing defaults and override precedence are retained, not globally unified.
The historical case fallbacks for tool/Direct thresholds remain2 seconds; the
current formal BrowseComp regression explicitly sets1/1. Only4P4D applies the
optional YAML profile at this case layer; other topologies do not silently gain
new profile semantics. This does not describe any downstream pipeline behavior.
Maintained versus historical replay entry classification remains a later task.

## Fail-closed parsing fixes

The profile function first resolves its path and captures the complete trusted
loader output. Only a successful loader may apply variables or select the
profile. Invalid later fields therefore cannot partially apply earlier fields;
errors propagate both to strict launchers and conditional function callers.
Existing explicit values, including empty values, still override profile values.
Origin labels remain shell-local. This is not general rollback for a later shell
assignment/export failure, nor a protocol for arbitrary replacement loaders.
The loader's schema, allowlist, scalar rules and relative path resolution are
unchanged. A real-wrapper negative test confirms invalid input stops before
creating its run directory, calling fingerprint validation or reaching pipeline
startup; the fingerprint stub fails safely if that contract regresses.

TCP configuration now checks Router and relay ports against all worker, bootstrap
and reverse-bootstrap listeners sharing the same explicithost_ip. Router/relay
co-location is determined by that IP, not only logical node IDs. Configurations
with overlapping same-IP listeners fail before launch; legal reuse on different
IPs remains permitted. This does not infer DNS aliases, multi-NIC identity or
physical machine equivalence. A same-node TCP probe must use one consistent
explicit routable IPv4. Runtime ownership/NUMA/port/process cleanup gates remain
mandatory and are not replaced by this input validation.

## Evidence and remaining acceptance

scripts/verify_case_configuration.py proves the entire four wrappers, shared
helper, profile function and unchanged loader against immutable frontend0ed608d,
plus exactly three reviewed expressions in the entire TCP launcher. It is part
of scripts/verify_snapshot.py; no changed file is simply exempted.

Retained tests use automatically located checkouts and immutable historical
wrapper/profile/parser fixtures, not the maintained code compared with itself.
They cover14 destructive mutations,29 actual Bash argument comparisons, profile
error/override/path/sequence cases and real parser collisions, including nine
unchanged valid model-family/TP plans. Argument tests deliberately stub fingerprint
validation and stop before pipeline execution; they do not prove native readiness
or GPU performance. The engine remains unchanged atd265f7946.

Full CPU928 records and real-kernel native1018 records pass with zero failures,
errors or skips. Independent final CODE GO confirms the unchanged eight-file
runtime bundle7568843bfc6fd495e3199ae1cda72db6e9f9f914fd38c3100e850a13ba109970
and all eight ownership/fault invariants. Actual H100 registered/bounce CUDA2
also pass. The new frozen BrowseComp comparison receives independent exact-plan
and final GO:300.973000+1200.001009 seconds, Decode11212.44 token/s,−0.47%
versus ORIGINALbaseline,100% parent reuse,0failures/OOM and clean final gates.
Its complete request and ownership audits are recorded with case-configuration-r1.
The latter exercises the extracted common argv settings with explicit1/1 values;
EXPERIMENT_CONFIG is unset, so it must not be reported as profile-driven GPU
coverage. Actual TCP/TP2/hybrid coverage and the full cleanup goal remain open.
