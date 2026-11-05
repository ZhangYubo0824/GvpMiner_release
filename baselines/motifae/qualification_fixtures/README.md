# Fixed Gibbs guard regression fixtures

These are user research inputs and output data, not the licensed GibbsCluster
program. Supply your own supported installation. The three installation hashes
remain fixed in `scripts/baseline_adapters/motifae_native_guard.py`.

`normal_reference/` is a byte-for-byte minimal copy of the successful original
GvpA pilot at `motifae_results/GvpA_gibbs_pilot_001/pilot_001/GvpA_feature000001_48244`:
`res/gibbs.1g.out`, `res/gibbs.1g.ds.out`, `cores/gibbs.1g.cores`, and
`data/pepdata.txt`. `normal_peptides.txt` is the original mapping-stage input
`motifae_results/discovery/GvpA/run_002/gibbs_inputs/peptides/feature_000001.txt`.
Its unique sequence set must equal the recorded native input.

`fault_peptides.txt` is the confirmed empty-single-cluster regression input:
23 instances of AIGSE and one SSAPQ. It is a software qualification fixture,
not a requirement on any family's data, feature index, or motif identifier.

`fixture_manifest.json` contains every data-file SHA-256. Its own SHA-256 is
fixed in the qualification code. A removed hash, edited input, changed native
reference, or changed inventory blocks qualification. This README is explanatory
and is not used as computational evidence.

On Linux x86_64 with the supported Gibbs installation and `gdb`, run:

```bash
python scripts/baseline_adapters/motifae_family_alignment.py qualify \
  --gibbs-root /path/to/gibbscluster-2.0 --output-dir /new/qualification
```

Qualification runs both fixtures. The normal case must use zero guard skips and
match full raw/display native G rows, ordered cores, all printed native parameter
settings, and the exact KLD. Only the BLOSUM installation-root spelling may be
normalized after verifying the installation's fixed BLOSUM content hash. The
fault case must record a positive guard counter and pass strict native parsing,
including exhaustive all-trash accounting if appropriate.

The saved qualification binds code, program hashes, and fixture hashes, not a
family FASTA or preparation. Guarded-first alignment separately verifies each
family's preparation and full FASTA before sampling its own peptide inputs.
Each eligible group is sampled once under the qualified runtime guard. No
result or mapping is reused merely by feature stem.

Windows unit tests use fixture output and explicitly mocked execution. They do
not create evidence of a real Linux sampler qualification. Generate the actual
qualification in the clean upload directory on the target Linux installation;
do not reuse a qualification from another code revision.
