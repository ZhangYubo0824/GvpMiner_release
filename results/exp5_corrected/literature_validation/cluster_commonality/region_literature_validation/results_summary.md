# Region-level motif validation summary

## Structural results

- Input: 14,782 high/moderate cluster-level secondary-structure commonality rows.
- Homologous experimental-region mapping: 3,368 rows.
- No region-specific literature mapping: 11,411 rows.
- Representative full sequence missing: 3 GvpJ rows.

The strongest structure-to-biology links are:

- **GvpA:** alpha1, beta-hairpin and alpha2 mappings connect conserved local
  structures to inter-rib bridging, rib-core hydrogen bonding, the hydrophobic
  gas-facing surface, gas-diffusion slits and GvpC-mediated reinforcement.
- **GvpC:** alpha-helical 33-aa repeat mappings support binding along GvpA ribs
  and mechanical reinforcement of the shell.
- **GvpF:** mapped motifs lie in the experimentally resolved two-domain
  alpha+beta fold. This supports a stable assembly scaffold/interaction-surface
  interpretation, but not a residue-level mechanism.
- **GvpJ/GvpM:** common structures map to mutation-sensitive or
  fragment-interaction regions. Their secondary structures are modeled rather
  than experimentally resolved, so the conclusion is region-function support,
  not an atomic structure assignment.
- **GvpL:** mapped motifs in residues homologous to L133-281 can be linked to
  GvpK interaction, but the tested fragment is broad.

## Functional results

- Input: 2,584 motifs with current clusterProfiler status
  `external_enrichment_not_significant` or `external_enrichment_no_terms`.
- Homologous local experimental-region support: 465 motifs.
- No localized support identified: 2,119 motifs.
- No functional evidence grades are assigned.

Supported motifs by family:

| Family | Motifs |
|---|---:|
| GvpJ | 315 |
| GvpL | 53 |
| GvpA | 42 |
| GvpM | 22 |
| GvpU | 18 |
| GvpC | 15 |

Interpretation:

- **GvpJ:** motifs overlap experimentally tested N-/C-terminal fragments,
  alpha1-loop substitutions, or the RAAIA/EYGL region associated with assembly,
  stability and GvpL interaction.
- **GvpL:** motifs map to the L133-281 fragment that retains GvpK interaction.
- **GvpA:** motifs map to A1-22 or A44-76 fragments tested for interactions
  with assembly proteins.
- **GvpM:** motifs map to the N-terminal 25-aa GvpL-binding region, the
  mutation-sensitive alpha1/GAV/RAAIA regions, or the C-terminal interaction
  fragment.
- **GvpU:** only motifs retaining the same experimentally mutated residue after
  alignment are retained; evidence concerns a five-site mutant and cannot assign
  causality to one residue.
- **GvpC:** motifs map to repeat deletions or conserved repeat-face residues
  tested for shell stabilization. Most supported GvpC clusters have low sequence
  counts and require cautious reporting.

## Important negative result

GvpK remains fully included in the external-enrichment audit. None of its 1,223
candidates could be mapped to a verified mutation, deletion, truncation or
fragment experiment on a local region of GvpK itself. Evidence that full-length
GvpK interacts with a GvpL fragment is not treated as GvpK motif evidence.

## Reporting boundary

`homologous_region_supported` means that a representative motif maps to a local
region experimentally tested in a homologous protein. It does not mean that the
exact representative accession or exact 30-aa window has been directly tested.
`no_localized_support` means that no qualifying local experiment was identified;
it does not imply absence of function.
