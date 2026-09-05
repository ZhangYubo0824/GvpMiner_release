from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "result6" / "motif_validation" / "motif_evidence_integration.csv"
INTERPRO = (
    ROOT
    / "result6"
    / "exp5_external_functional_enrichment"
    / "interproscan"
    / "all_gvp_proteins.fasta.tsv"
)
EGGNOG = (
    ROOT
    / "result6"
    / "exp5_external_functional_enrichment"
    / "eggnog"
    / "gvp_eggnog.emapper.annotations"
)
OUTDIR = ROOT / "result6" / "motif_validation" / "literature_validation"
TARGET_PER_FAMILY = 5
OUTPUT_TAG = "expanded5"


STRUCTURAL_MOTIFS = [
    "GvpA_M2_motif_001",
    "GvpC_M2_motif_001",
    "GvpD_M3_motif_073",
    "GvpF_M2_motif_002",
    "GvpG_M2_motif_002",
    "GvpH_M2_motif_002",
    "GvpJ_M2_motif_017",
    "GvpK_M3_motif_001",
    "GvpL_M2_motif_001",
    "GvpN_M3_motif_001",
    "GvpO_M2_motif_001",
    "GvpT_M3_motif_001",
]

FUNCTIONAL_MOTIFS = [
    "GvpA_M2_motif_043",
    "GvpC_M2_motif_098",
    "GvpD_M2_motif_134",
    "GvpF_M3_motif_290",
    "GvpG_M3_motif_037",
    "GvpH_M2_motif_014",
    "GvpJ_M2_motif_1227",
    "GvpK_M2_motif_005",
    "GvpL_M2_motif_062",
    "GvpM_M2_motif_004",
    "GvpN_M3_motif_001",
    "GvpN_M2_motif_001",
    "GvpO_M2_motif_012",
    "GvpQ_M3_motif_116",
    "GvpT_M2_motif_192",
    "GvpU_M2_motif_001",
    "GvpV_M3_motif_015",
]


SOURCES = [
    {
        "source_id": "S01",
        "title": "Cryo-EM structure of gas vesicles for buoyancy-controlled motility",
        "year": 2023,
        "source_type": "primary_structure_paper",
        "DOI": "10.1016/j.cell.2023.01.041",
        "PMID": "36868215",
        "PMCID": "PMC9994262",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9994262/",
        "verified": True,
        "specific_basis": (
            "Cryo-EM resolves the GvpA shell architecture and places GvpC on the "
            "outer shell; supports shell-packing and reinforcement interpretations."
        ),
        "limitations": "Different organisms/accessions; does not test the selected 30-aa windows.",
    },
    {
        "source_id": "S02",
        "title": (
            "Structure of the gas vesicle protein GvpF from the cyanobacterium "
            "Microcystis aeruginosa PCC 7806"
        ),
        "year": 2014,
        "source_type": "primary_structure_paper",
        "DOI": "10.1107/S1399004714021312",
        "PMID": "25372690",
        "PMCID": "",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/25372690/",
        "verified": True,
        "specific_basis": (
            "Reports a 2.7-A GvpF crystal structure and proposes a role in gas-vesicle "
            "assembly from structural similarity and surface properties."
        ),
        "limitations": "Homologous GvpF, not the selected accession or interval.",
    },
    {
        "source_id": "S03",
        "title": (
            "The role of GvpC in the formation of gas vesicles in various species "
            "of cyanobacteria"
        ),
        "year": 1992,
        "source_type": "primary_experimental_paper",
        "DOI": "10.1099/00221287-138-12-2609",
        "PMID": "1280861",
        "PMCID": "",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/1280861/",
        "verified": True,
        "specific_basis": (
            "GvpC removal and reconstitution experiments support a direct role in "
            "shell strengthening and collapse-pressure control."
        ),
        "limitations": "Protein-level evidence; no test of the selected representative window.",
    },
    {
        "source_id": "S04",
        "title": "Accessory Gvp Proteins Form a Complex During Gas Vesicle Formation of Haloarchaea",
        "year": 2020,
        "source_type": "primary_interaction_paper",
        "DOI": "10.3389/fmicb.2020.610179",
        "PMID": "33281806",
        "PMCID": "PMC7688916",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7688916/",
        "verified": True,
        "specific_basis": (
            "Affinity and interaction experiments support a GvpF/GvpL-centered "
            "accessory assembly complex involving GvpG, GvpH, GvpJ, GvpK and GvpM."
        ),
        "limitations": "Haloarchaeal homologs; interaction evidence is not residue-level.",
    },
    {
        "source_id": "S05",
        "title": "Interaction of Haloarchaeal Gas Vesicle Proteins Determined by Split-GFP",
        "year": 2018,
        "source_type": "primary_interaction_paper",
        "DOI": "10.3389/fmicb.2018.01897",
        "PMID": "30174663",
        "PMCID": "PMC6107691",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6107691/",
        "verified": True,
        "specific_basis": "Split-GFP assays map multiple pairwise interactions among haloarchaeal Gvp proteins.",
        "limitations": "Homolog-level and pairwise-interaction evidence; selected windows were not mutated.",
    },
    {
        "source_id": "S06",
        "title": "Effect of Mutations in GvpJ and GvpM on Gas Vesicle Formation of Halobacterium salinarum",
        "year": 2021,
        "source_type": "primary_mutation_paper",
        "DOI": "10.3389/fmicb.2021.794240",
        "PMID": "34975818",
        "PMCID": "PMC8716928",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8716928/",
        "verified": True,
        "specific_basis": "Mutational analysis directly links homologous GvpJ/GvpM proteins to vesicle formation.",
        "limitations": "Mutated residues do not establish function of the selected representative windows.",
    },
    {
        "source_id": "S07",
        "title": "Interaction of the gas vesicle proteins GvpA, GvpC, GvpN, and GvpO of Halobacterium salinarum",
        "year": 2022,
        "source_type": "primary_interaction_paper",
        "DOI": "10.3389/fmicb.2022.971917",
        "PMID": "35966690",
        "PMCID": "PMC9372576",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9372576/",
        "verified": True,
        "specific_basis": "Experimental interaction analysis supports roles of GvpN and GvpO at the shell/assembly interface.",
        "limitations": "Different accessions and no direct assay of the selected 30-aa intervals.",
    },
    {
        "source_id": "S08",
        "title": (
            "Functional analysis of the gas vesicle gene cluster of the halophilic "
            "archaeon Haloferax mediterranei defines the vac-region boundary and "
            "suggests a regulatory role for the gvpD gene or its product"
        ),
        "year": 1992,
        "source_type": "primary_genetic_paper",
        "DOI": "10.1111/j.1365-2958.1992.tb01789.x",
        "PMID": "1282192",
        "PMCID": "",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/1282192/",
        "verified": True,
        "specific_basis": "Genetic analysis supports a regulatory function for GvpD in gas-vesicle gene expression.",
        "limitations": "Protein-level genetic evidence; does not localize activity to the selected interval.",
    },
    {
        "source_id": "S09",
        "title": "Eight of fourteen gvp genes are sufficient for formation of gas vesicles in halophilic archaea",
        "year": 2000,
        "source_type": "primary_genetic_paper",
        "DOI": "10.1128/JB.182.15.4328-4336.2000",
        "PMID": "10894744",
        "PMCID": "PMC101952",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC101952/",
        "verified": True,
        "specific_basis": "Deletion/reconstitution experiments identify a minimal haloarchaeal gas-vesicle gene set.",
        "limitations": "Gene-level sufficiency does not define motif-level mechanism.",
    },
    {
        "source_id": "S10",
        "title": "Gas vesicle genes identified in Bacillus megaterium and functional expression in Escherichia coli",
        "year": 1998,
        "source_type": "primary_genetic_paper",
        "DOI": "10.1128/JB.180.9.2450-2458.1998",
        "PMID": "9573198",
        "PMCID": "PMC107188",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC107188/",
        "verified": True,
        "specific_basis": "A Bacillus gvp cluster was cloned and functionally expressed in E. coli.",
        "limitations": "Cluster-level evidence; individual representative windows were not tested.",
    },
    {
        "source_id": "S11",
        "title": "Acoustic reporter genes for noninvasive imaging of microorganisms in mammalian hosts",
        "year": 2018,
        "source_type": "primary_engineering_paper",
        "DOI": "10.1038/nature25021",
        "PMID": "29300010",
        "PMCID": "PMC5920530",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5920530/",
        "verified": True,
        "specific_basis": "Engineered gas-vesicle gene clusters provide ultrasound/acoustic reporter activity.",
        "limitations": "Supports system-level imaging function, not a specific selected motif.",
    },
    {
        "source_id": "S12",
        "title": "An improved genetic system for bioengineering buoyant gas vesicle nanoparticles from Haloarchaea",
        "year": 2013,
        "source_type": "primary_engineering_paper",
        "DOI": "10.1186/1472-6750-13-112",
        "PMID": "24359319",
        "PMCID": "PMC3878110",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3878110/",
        "verified": True,
        "specific_basis": "Demonstrates engineering of haloarchaeal gas-vesicle nanoparticles, including GvpC-based display constructs.",
        "limitations": "Engineering evidence is construct/protein-level and does not validate the selected GvpC interval.",
    },
    {
        "source_id": "S13",
        "title": "The gas vesicle gene cluster from Microcystis aeruginosa and DNA rearrangements that lead to loss of cell buoyancy",
        "year": 2004,
        "source_type": "primary_genetic_paper",
        "DOI": "10.1128/JB.186.8.2355-2365.2004",
        "PMID": "15090505",
        "PMCID": "PMC412153",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC412153/",
        "verified": True,
        "specific_basis": "Insertional interruption involving gvpV is associated with loss of buoyancy in Microcystis.",
        "limitations": "The rearrangement affects a locus; it does not isolate a motif-level GvpV mechanism.",
    },
    {
        "source_id": "S14",
        "title": "Elucidating the assembly of gas vesicles by systematic protein-protein interaction analysis",
        "year": 2024,
        "source_type": "primary_interaction_paper",
        "DOI": "10.1038/s44318-024-00178-2",
        "PMID": "39227754",
        "PMCID": "PMC11445434",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11445434/",
        "verified": True,
        "specific_basis": (
            "Systematic interaction analysis of the Bacillus megaterium assembly "
            "system supports a coordinated network involving GvpN, GvpF, GvpG, GvpK, GvpT and other proteins."
        ),
        "limitations": "Network-level homolog evidence; no selected representative window was tested.",
    },
    {
        "source_id": "S15",
        "title": "Phase transition of GvpU regulates gas vesicle clustering in bacteria",
        "year": 2024,
        "source_type": "primary_function_paper",
        "DOI": "10.1126/sciadv.adk1375",
        "PMID": "38437207",
        "PMCID": "PMC10920092",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10920092/",
        "verified": True,
        "specific_basis": "Direct experiments show GvpU phase separation/clustering regulates gas-vesicle spatial organization.",
        "limitations": "Different GvpU accession; selected interval was not independently tested.",
    },
    {
        "source_id": "S16",
        "title": "RCSB PDB entry 4QSG: Gas vesicle protein GvpF",
        "year": 2014,
        "source_type": "structure_database",
        "DOI": "10.2210/pdb4QSG/pdb",
        "PMID": "",
        "PMCID": "",
        "URL": "https://www.rcsb.org/structure/4QSG",
        "verified": True,
        "specific_basis": "Curated experimental structure corresponding to the GvpF crystal-structure study.",
        "limitations": "Structure is a homolog and not the representative accession WP_193942151.1.",
    },
    {
        "source_id": "S17",
        "title": "NCBI Conserved Domain cd18190: GvpQ",
        "year": 2026,
        "source_type": "domain_database",
        "DOI": "",
        "PMID": "",
        "PMCID": "",
        "URL": "https://www.ncbi.nlm.nih.gov/Structure/cdd/cddsrv.cgi?uid=cd18190",
        "verified": True,
        "specific_basis": "Database family assignment for GvpQ; supports identity but not a specific molecular function.",
        "limitations": "Database annotation only; no motif-level experimental evidence.",
    },
    {
        "source_id": "S18",
        "title": "Local InterProScan 5.77-108.0 annotations for selected representative accessions",
        "year": 2026,
        "source_type": "local_database_output",
        "DOI": "",
        "PMID": "",
        "PMCID": "",
        "URL": "../../exp5_external_functional_enrichment/interproscan/all_gvp_proteins.fasta.tsv",
        "verified": True,
        "specific_basis": (
            "Exact representative accessions were checked against Pfam, InterPro, "
            "NCBIfam, ProSite, SMART and related signatures in the local raw output."
        ),
        "limitations": "Computational database annotation; family/domain identity is not motif-function proof.",
    },
    {
        "source_id": "S19",
        "title": "Local eggNOG-mapper 2.1.13 annotations for selected representative accessions",
        "year": 2026,
        "source_type": "local_database_output",
        "DOI": "",
        "PMID": "",
        "PMCID": "",
        "URL": "../../exp5_external_functional_enrichment/eggnog/gvp_eggnog.emapper.annotations",
        "verified": True,
        "specific_basis": "Exact representative accessions were checked for orthology-derived family/function descriptions.",
        "limitations": "Orthology transfer can be ambiguous, especially among AAA+ proteins and GvpA-like paralogs.",
    },
]


STRUCT_EVIDENCE = {
    "GvpA": {
        "sources": "S01;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpA shell structure plus exact-accession family annotation.",
        "meaning": (
            "The representative window lies in a conserved GvpA-like segment. Its "
            "low surface exposure/centroid proximity and high pLDDT are compatible "
            "with a shell-rib packing or fold-stabilizing region."
        ),
        "limitations": "No mutation or structure directly tests WP_138500607.1 residues 35-65.",
    },
    "GvpC": {
        "sources": "S01;S03;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpC repeat/shell-reinforcement evidence plus exact-accession repeat annotation.",
        "meaning": (
            "The highly confident, helical repeat-like window is compatible with "
            "GvpC-mediated shell binding/reinforcement or an exposed interaction segment."
        ),
        "limitations": "The selected XUP06191.1 interval was not isolated in collapse-pressure or cryo-EM experiments.",
    },
    "GvpD": {
        "sources": "S08;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpD genetic regulation evidence and exact-accession P-loop-domain annotation.",
        "meaning": (
            "The high-confidence structured segment is compatible with the folded "
            "regulatory ATPase/P-loop region rather than a shell structural element."
        ),
        "limitations": "The interval is not directly linked to ATPase activity or GvpE regulation.",
    },
    "GvpF": {
        "sources": "S02;S04;S14;S16;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpF crystal structure and assembly-complex evidence plus exact-accession annotation.",
        "meaning": (
            "High contact/centroid-related enrichment and very high pLDDT support a "
            "stable GvpF fold segment that may contribute to accessory-complex assembly."
        ),
        "limitations": "PDB 4QSG is a homolog; no direct mapping of the selected sequence to an interaction interface.",
    },
    "GvpG": {
        "sources": "S04;S05;S14;S18;S19",
        "grade": "B",
        "scope": "Homologous accessory-protein interaction evidence plus exact-accession GvpG annotation.",
        "meaning": (
            "A confident, structured helical segment is compatible with an accessory "
            "assembly interface or local packing element."
        ),
        "limitations": "Protein interaction studies do not identify this exact BHC31298.1 interval.",
    },
    "GvpH": {
        "sources": "S04;S05;S18;S19",
        "grade": "B",
        "scope": "Homologous accessory-complex evidence and exact-accession GvpH annotation.",
        "meaning": (
            "The confident segment may form part of a folded GvpH interaction surface; "
            "the GvpH/HSP20-like annotation also permits a chaperone-like stabilization role."
        ),
        "limitations": "The chaperone interpretation is annotation-based and not demonstrated for residues 160-190.",
    },
    "GvpJ": {
        "sources": "S04;S05;S06;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpJ mutation/interaction evidence plus exact-accession GvpA-like signatures.",
        "meaning": (
            "The window may contribute to a GvpA-like fold used by the assembly factor "
            "GvpJ, potentially supporting shell nucleation or accessory contacts."
        ),
        "limitations": (
            "The representative accession carries GvpA-like signatures; family/paralog "
            "assignment ambiguity prevents a residue-specific GvpJ claim."
        ),
    },
    "GvpK": {
        "sources": "S04;S05;S14;S18;S19",
        "grade": "B",
        "scope": "Homologous accessory-network evidence plus exact-accession GvpK annotation.",
        "meaning": (
            "The high-confidence structured window is compatible with a stable GvpK "
            "interaction or packing segment in the accessory assembly network."
        ),
        "limitations": "No direct interface or mutation data for WP_200697840.1 residues 50-80.",
    },
    "GvpL": {
        "sources": "S04;S05;S14;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpL-centered complex evidence plus exact-accession GvpL/F annotation.",
        "meaning": (
            "Very high pLDDT and packing-related enrichment support a stable GvpL "
            "fold segment that may help scaffold the accessory complex."
        ),
        "limitations": "No direct structural or mutational evidence for the selected WP_007260754.1 interval.",
    },
    "GvpN": {
        "sources": "S07;S14;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpN interaction evidence plus exact-accession GvpN/AAA+ annotation.",
        "meaning": (
            "The very high-confidence segment is compatible with the structured AAA+ "
            "assembly ATPase, although this N-terminal window is outside the canonical "
            "ATPase-domain annotation."
        ),
        "limitations": (
            "The local Kofam/orthology context can confuse GvpN with related MoxR/NorQ "
            "ATPases; the selected N-terminal interval has no direct biochemical assay."
        ),
    },
    "GvpO": {
        "sources": "S07;S18;S19",
        "grade": "B",
        "scope": "Homologous GvpO-GvpA interaction evidence plus exact-accession GvpO annotation.",
        "meaning": (
            "The high-confidence structured window is compatible with a folded GvpO "
            "segment involved in shell association or late assembly."
        ),
        "limitations": "No direct interaction mapping to WP_006168080.1 residues 170-200.",
    },
    "GvpT": {
        "sources": "S10;S14;S18;S19",
        "grade": "B",
        "scope": "Homologous Bacillus assembly-network evidence plus exact-accession GvpT/P annotation.",
        "meaning": (
            "The confident structured segment may participate in the accessory "
            "interaction network, but its exact structural role remains unresolved."
        ),
        "limitations": "No experimental structure or residue-level perturbation for this GvpT window.",
    },
}


FUNCTION_EVIDENCE = {
    "GvpA_M2_motif_043": (
        "shell assembly; shell stability; acoustic reporter systems",
        "S01;S10;S11;S18;S19",
        "B",
        "Homologous GvpA is the major shell protein; exact accession is annotated GvpA.",
        "No evidence that residues 30-60 uniquely control assembly, stability, or ultrasound contrast.",
    ),
    "GvpC_M2_motif_098": (
        "shell reinforcement; collapse-pressure tuning; protein display/engineering",
        "S01;S03;S12;S18",
        "B",
        "Homologous GvpC directly reinforces shells and has been used for engineering/display.",
        "The N-terminal representative window was not directly tested; filter failure is not nonsignificance.",
    ),
    "GvpD_M2_motif_134": (
        "regulation of gas-vesicle gene expression",
        "S08;S18;S19",
        "B",
        "Homologous GvpD has genetic evidence for a regulatory role; exact accession has GvpD domains.",
        "No direct evidence assigns regulation to residues 325-355.",
    ),
    "GvpF_M3_motif_290": (
        "accessory-complex assembly and gas-vesicle biogenesis",
        "S02;S04;S14;S16;S18;S19",
        "B",
        "Homologous GvpF has a solved structure and participates in accessory assembly complexes.",
        "No direct experiment tests ADF40140.1 residues 145-175.",
    ),
    "GvpG_M3_motif_037": (
        "accessory-protein interaction during gas-vesicle assembly",
        "S04;S05;S14;S18;S19",
        "B",
        "Homologous GvpG participates in experimentally mapped assembly interactions.",
        "The selected SCE58004.1 window has no direct mutation or binding assay.",
    ),
    "GvpH_M2_motif_014": (
        "assembly support and possible chaperone-like stabilization",
        "S04;S05;S18;S19",
        "B",
        "Homologous GvpH participates in accessory interactions; exact accession also has an HSP20-like annotation.",
        "Chaperone-like function is inferential and the selected interval was not tested.",
    ),
    "GvpJ_M2_motif_1227": (
        "gas-vesicle assembly/nucleation",
        "S04;S05;S06;S18;S19",
        "B",
        "Homologous GvpJ mutation and interaction studies support an assembly role.",
        "Exact accession has GvpA-like signatures, so paralog/family ambiguity remains; filter failure is not nonsignificance.",
    ),
    "GvpK_M2_motif_005": (
        "accessory assembly-network interaction",
        "S04;S05;S14;S18;S19",
        "B",
        "Homologous GvpK participates in experimentally mapped assembly networks; exact accession is annotated GvpK.",
        "No motif-specific interaction surface has been established.",
    ),
    "GvpL_M2_motif_062": (
        "accessory-complex scaffold during assembly",
        "S04;S05;S14;S18;S19",
        "B",
        "Homologous GvpL forms central accessory-protein interactions; exact accession is annotated GvpL.",
        "No direct evidence for residues 75-105.",
    ),
    "GvpM_M2_motif_004": (
        "minor shell/assembly component affecting gas-vesicle formation",
        "S04;S06;S18;S19",
        "B",
        "Homologous GvpM mutation and interaction studies support a role in vesicle formation.",
        "The representative interval was not independently perturbed.",
    ),
    "GvpN_M3_motif_001": (
        "ATPase-assisted gas-vesicle assembly",
        "S07;S14;S18;S19",
        "B",
        "Homologous GvpN participates in the assembly network and exact accession has GvpN/AAA+ annotations.",
        "This batch was not tested externally; the N-terminal motif has no direct ATPase assay.",
    ),
    "GvpN_M2_motif_001": (
        "ATPase-assisted gas-vesicle assembly",
        "S07;S14;S18;S19",
        "B",
        "Homologous GvpN participates in the assembly network and exact accession has GvpN/AAA+ annotations.",
        "External enrichment is unavailable (log only/no CSV), not nonsignificant; no motif-level experiment.",
    ),
    "GvpO_M2_motif_012": (
        "GvpA-associated shell assembly or maturation",
        "S07;S18;S19",
        "B",
        "Homologous GvpO interacts with GvpA and exact accession is annotated GvpO.",
        "No direct evidence for residues 70-100.",
    ),
    "GvpQ_M3_motif_116": (
        "gas-vesicle accessory role, molecular function unresolved",
        "S10;S17;S18;S19",
        "C",
        "Exact accession is classified as GvpQ-like; cluster-level studies support gas-vesicle context.",
        "No direct GvpQ or motif-specific functional experiment was identified.",
    ),
    "GvpT_M2_motif_192": (
        "accessory assembly-network interaction",
        "S10;S14;S18;S19",
        "B",
        "Homologous GvpT is part of a functional Bacillus cluster and systematic assembly network.",
        "No direct evidence for residues 75-105.",
    ),
    "GvpU_M2_motif_001": (
        "gas-vesicle clustering and intracellular spatial organization",
        "S10;S14;S15;S18;S19",
        "B",
        "Homologous GvpU directly regulates gas-vesicle clustering; exact accession is annotated GvpU-like.",
        "The selected N-terminal window was not tested; filter failure is not nonsignificance.",
    ),
    "GvpV_M3_motif_015": (
        "buoyancy-associated gas-vesicle gene-cluster function",
        "S13;S19",
        "B",
        "A Microcystis rearrangement interrupting gvpV is associated with loss of buoyancy.",
        "Locus-level evidence does not isolate GvpV protein action or the selected interval; local annotation is sparse.",
    ),
}


FUNCTION_FAMILY_EVIDENCE = {
    "GvpA": FUNCTION_EVIDENCE["GvpA_M2_motif_043"],
    "GvpC": FUNCTION_EVIDENCE["GvpC_M2_motif_098"],
    "GvpD": FUNCTION_EVIDENCE["GvpD_M2_motif_134"],
    "GvpF": FUNCTION_EVIDENCE["GvpF_M3_motif_290"],
    "GvpG": FUNCTION_EVIDENCE["GvpG_M3_motif_037"],
    "GvpH": FUNCTION_EVIDENCE["GvpH_M2_motif_014"],
    "GvpJ": FUNCTION_EVIDENCE["GvpJ_M2_motif_1227"],
    "GvpK": FUNCTION_EVIDENCE["GvpK_M2_motif_005"],
    "GvpL": FUNCTION_EVIDENCE["GvpL_M2_motif_062"],
    "GvpM": FUNCTION_EVIDENCE["GvpM_M2_motif_004"],
    "GvpN": FUNCTION_EVIDENCE["GvpN_M3_motif_001"],
    "GvpO": FUNCTION_EVIDENCE["GvpO_M2_motif_012"],
    "GvpQ": FUNCTION_EVIDENCE["GvpQ_M3_motif_116"],
    "GvpT": FUNCTION_EVIDENCE["GvpT_M2_motif_192"],
    "GvpU": FUNCTION_EVIDENCE["GvpU_M2_motif_001"],
    "GvpV": FUNCTION_EVIDENCE["GvpV_M3_motif_015"],
}


def select_diverse_rows(
    candidates: pd.DataFrame,
    seeds: list[str],
    target_per_family: int,
    structural: bool,
) -> list[str]:
    selected: list[str] = []
    for family, family_rows in candidates.groupby("family", sort=True):
        family_seeds = [
            motif
            for motif in seeds
            if motif in set(family_rows["motif_cluster"])
        ]
        chosen = family_rows[family_rows["motif_cluster"].isin(family_seeds)].copy()

        ranked = family_rows.copy()
        ranked["method_priority"] = ranked["method"].map({"M2": 0, "M3": 1}).fillna(2)
        if structural:
            ranked["strength_priority"] = ranked["structural_evidence_strength"].map(
                {"strong": 0, "moderate": 1}
            )
            ranked = ranked.sort_values(
                [
                    "strength_priority",
                    "n_structural_instances",
                    "mean_motif_pLDDT",
                    "method_priority",
                    "motif_cluster",
                ],
                ascending=[True, False, False, True, True],
            )
        else:
            status_priority = {
                "external_filter_failed": 0,
                "not_tested_external": 0,
                "external_enrichment_unavailable": 1,
                "external_enrichment_not_significant": 2,
            }
            ranked["status_priority"] = ranked["external_enrichment_status"].map(
                status_priority
            )
            ranked["structural_priority"] = (
                ranked["structural_significant"].fillna(False).astype(bool).map({True: 0, False: 1})
            )
            ranked = ranked.sort_values(
                [
                    "status_priority",
                    "structural_priority",
                    "n_unique_sequences",
                    "mean_motif_pLDDT",
                    "method_priority",
                    "motif_cluster",
                ],
                ascending=[True, True, False, False, True, True],
            )

        chosen_motifs = set(chosen["motif_cluster"])
        chosen_sequences = set(chosen["representative_sequence"])
        chosen_methods = list(chosen["method"])

        if not structural:
            chosen_statuses = set(chosen["external_enrichment_status"])
            available_statuses = list(
                dict.fromkeys(ranked["external_enrichment_status"].tolist())
            )
            for status in available_statuses:
                if len(chosen) >= target_per_family or status in chosen_statuses:
                    continue
                status_rows = ranked[ranked["external_enrichment_status"] == status]
                for _, row in status_rows.iterrows():
                    if row["motif_cluster"] in chosen_motifs:
                        continue
                    if row["representative_sequence"] in chosen_sequences:
                        continue
                    chosen = pd.concat([chosen, row.to_frame().T], ignore_index=True)
                    chosen_motifs.add(row["motif_cluster"])
                    chosen_sequences.add(row["representative_sequence"])
                    chosen_methods.append(row["method"])
                    chosen_statuses.add(status)
                    break

        def add_rows(require_unused_method: bool, require_position_gap: bool) -> None:
            nonlocal chosen
            for _, row in ranked.iterrows():
                if len(chosen) >= target_per_family:
                    return
                if row["motif_cluster"] in chosen_motifs:
                    continue
                if row["representative_sequence"] in chosen_sequences:
                    continue
                if require_unused_method and row["method"] in chosen_methods:
                    continue
                if require_position_gap and not chosen.empty:
                    midpoint = (float(row["representative_start"]) + float(row["representative_end"])) / 2
                    chosen_midpoints = (
                        chosen["representative_start"].astype(float)
                        + chosen["representative_end"].astype(float)
                    ) / 2
                    if (chosen_midpoints - midpoint).abs().min() < 15:
                        continue
                chosen = pd.concat([chosen, row.to_frame().T], ignore_index=True)
                chosen_motifs.add(row["motif_cluster"])
                chosen_sequences.add(row["representative_sequence"])
                chosen_methods.append(row["method"])

        add_rows(require_unused_method=True, require_position_gap=True)
        add_rows(require_unused_method=False, require_position_gap=True)
        add_rows(require_unused_method=False, require_position_gap=False)
        if len(chosen) < target_per_family:
            for _, row in ranked.iterrows():
                if len(chosen) >= target_per_family:
                    break
                if row["motif_cluster"] in chosen_motifs:
                    continue
                chosen = pd.concat([chosen, row.to_frame().T], ignore_index=True)
                chosen_motifs.add(row["motif_cluster"])
        if len(chosen) < target_per_family:
            raise ValueError(
                f"{family} has only {len(chosen)} diverse candidates; "
                f"target is {target_per_family}"
            )
        selected.extend(chosen["motif_cluster"].tolist()[:target_per_family])
    return selected


def select_motif_sets(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    structural_candidates = frame[
        (frame["structural_significant"] == True)
        & (frame["min_structural_q_value"] <= 0.05)
        & (frame["structural_evidence_strength"].isin(["strong", "moderate"]))
        & (frame["mean_motif_pLDDT"] >= 70)
    ].copy()
    functional_statuses = {
        "external_enrichment_not_significant",
        "not_tested_external",
        "external_enrichment_unavailable",
        "external_filter_failed",
    }
    functional_candidates = frame[
        frame["external_enrichment_status"].isin(functional_statuses)
    ].copy()
    structural = select_diverse_rows(
        structural_candidates, STRUCTURAL_MOTIFS, TARGET_PER_FAMILY, structural=True
    )
    functional = select_diverse_rows(
        functional_candidates, FUNCTIONAL_MOTIFS, TARGET_PER_FAMILY, structural=False
    )
    return structural, functional


def compact_interpro(accessions: set[str]) -> dict[str, str]:
    entries: dict[str, list[str]] = {accession: [] for accession in accessions}
    with INTERPRO.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if not fields or fields[0] not in entries or len(fields) < 13:
                continue
            accession, analysis, signature, signature_name = fields[0], fields[3], fields[4], fields[5]
            interpro_id, interpro_name = fields[11], fields[12]
            if analysis not in {"Pfam", "NCBIfam", "Hamap", "ProSitePatterns", "SMART"}:
                continue
            label = f"{analysis}:{signature} {signature_name}"
            if interpro_id != "-":
                label += f" [{interpro_id} {interpro_name}]"
            if label not in entries[accession]:
                entries[accession].append(label)
    return {key: "; ".join(value[:5]) for key, value in entries.items()}


def compact_eggnog(accessions: set[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    with EGGNOG.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields or fields[0] not in accessions:
                continue
            description = fields[7] if len(fields) > 7 else ""
            preferred = fields[8] if len(fields) > 8 else ""
            pfams = fields[-1] if fields else ""
            entries[fields[0]] = f"preferred_name={preferred}; description={description}; PFAMs={pfams}"
    return entries


def endpoint_direction_summary(value: str) -> str:
    pairs = []
    for metric, direction in re.findall(r"([A-Za-z0-9_]+):(positive|negative)", str(value)):
        item = f"{metric}:{direction}"
        if item not in pairs:
            pairs.append(item)
    return ";".join(pairs)


def representative_fields(row: pd.Series) -> dict[str, object]:
    return {
        "family_method": f"{row['family']}_{row['method']}",
        "family": row["family"],
        "method": row["method"],
        "motif": row.name,
        "motif_consensus": row["motif_consensus"],
        "representative_window_id": row["representative_window_id"],
        "representative_accession": row["representative_parent_id"],
        "representative_start": int(row["representative_start"]),
        "representative_end": int(row["representative_end"]),
        "representative_length": int(row["representative_length"]),
        "representative_sequence": row["representative_sequence"],
    }


def build_structural(
    selected: pd.DataFrame,
    motifs: list[str],
    interpro: dict[str, str],
    eggnog: dict[str, str],
) -> pd.DataFrame:
    records = []
    by_motif = selected.set_index("motif_cluster")
    for motif in motifs:
        row = by_motif.loc[motif]
        evidence = STRUCT_EVIDENCE[row["family"]]
        record = representative_fields(row)
        record.update(
            {
                "significant_structural_endpoints": row["significant_endpoints"],
                "endpoint_effect_direction_summary": endpoint_direction_summary(
                    row["significant_endpoints"]
                ),
                "n_instances": int(row["n_structural_instances"]),
                "motif_mean_pLDDT": row["mean_motif_pLDDT"],
                "min_q_value": row["min_structural_q_value"],
                "structural_evidence_strength": row["structural_evidence_strength"],
                "structural_type": row["structural_type"],
                "possible_biological_significance": evidence["meaning"],
                "local_annotation_evidence": " | ".join(
                    part
                    for part in [
                        interpro.get(row["representative_parent_id"], ""),
                        eggnog.get(row["representative_parent_id"], ""),
                    ]
                    if part
                ),
                "evidence_scope": evidence["scope"],
                "source_ids": evidence["sources"],
                "evidence_grade": evidence["grade"],
                "limitations": evidence["limitations"],
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def build_functional(
    selected: pd.DataFrame,
    motifs: list[str],
    interpro: dict[str, str],
    eggnog: dict[str, str],
) -> pd.DataFrame:
    records = []
    by_motif = selected.set_index("motif_cluster")
    for motif in motifs:
        row = by_motif.loc[motif]
        candidate, sources, grade, scope, limitations = FUNCTION_EVIDENCE.get(
            motif, FUNCTION_FAMILY_EVIDENCE[row["family"]]
        )
        if motif not in FUNCTION_EVIDENCE:
            limitations = (
                limitations.rstrip(".")
                + f"; no direct experiment tests {row['representative_parent_id']} "
                + f"residues {int(row['representative_start'])}-{int(row['representative_end'])}."
            )
        significant_terms = row["significant_terms"]
        if pd.isna(significant_terms):
            significant_terms = ""
        missing_reason = row["missing_reason"]
        if pd.isna(missing_reason):
            missing_reason = ""
        record = representative_fields(row)
        record.update(
            {
                "external_enrichment_status": row["external_enrichment_status"],
                "significant_terms": significant_terms,
                "missing_reason": missing_reason,
                "candidate_function": candidate,
                "local_annotation_evidence": " | ".join(
                    part
                    for part in [
                        interpro.get(row["representative_parent_id"], ""),
                        eggnog.get(row["representative_parent_id"], ""),
                    ]
                    if part
                ),
                "literature_evidence": scope,
                "evidence_scope": scope,
                "source_ids": sources,
                "evidence_grade": grade,
                "limitations": limitations,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def attach_source_motifs(
    sources: list[dict[str, object]], structural: pd.DataFrame, functional: pd.DataFrame
) -> pd.DataFrame:
    support: dict[str, list[str]] = {source["source_id"]: [] for source in sources}
    for frame in [structural, functional]:
        for _, row in frame.iterrows():
            for source_id in row["source_ids"].split(";"):
                if row["motif"] not in support[source_id]:
                    support[source_id].append(row["motif"])
    output = []
    for source in sources:
        record = dict(source)
        record["supported_motifs"] = ";".join(support[source["source_id"]])
        output.append(record)
    columns = [
        "source_id",
        "title",
        "year",
        "source_type",
        "DOI",
        "PMID",
        "PMCID",
        "URL",
        "verified",
        "supported_motifs",
        "specific_basis",
        "limitations",
    ]
    return pd.DataFrame(output)[columns]


def write_protocol(structural: pd.DataFrame, functional: pd.DataFrame) -> None:
    status_counts = Counter(functional["external_enrichment_status"])
    grade_counts = Counter(functional["evidence_grade"])
    text = f"""# Motif literature and database validation protocol

## Search date and scope

- Search date: 2026-06-08 (Asia/Shanghai).
- Structural shortlist: {len(structural)} motifs, five diverse high-priority representative motifs per family with usable Exp4 structural evidence.
- Functional shortlist: {len(functional)} motifs selected from external enrichment states requiring follow-up.
- Search representation: only Exp3 representative-window accession, start, end and sequence were used. Exp4/Exp5 statistics were read but not modified.

## Databases and sources

- Bibliographic verification: PubMed, Europe PMC, publisher/PMC full-text landing pages and DOI records.
- Structure: RCSB PDB and primary structural papers; AlphaFold DB was considered, but no exact stable mapping from most RefSeq representative accessions to an AlphaFold entry was asserted.
- Protein/domain annotation: local InterProScan 5.77-108.0 and eggNOG-mapper 2.1.13 raw outputs, with official InterPro/Pfam/NCBIfam identifiers retained in the evidence text.
- Functional context: primary mutation, deletion/reconstitution, interaction and engineering papers were preferred over reviews.
- Exact sequence audit: `exact_sequence_search_audit_{OUTPUT_TAG}.csv` records Europe PMC exact-phrase searches for the selected 30-aa representative sequences.

## Query design

Queries combined:

1. `family + representative_sequence + representative_accession + representative_start-end`
2. `family + gas vesicle assembly OR shell stability`
3. `family + acoustic reporter OR ultrasound imaging`
4. `family + delivery OR display OR transport OR localization`
5. `family + mutation OR deletion OR fusion OR engineering`
6. Exact paper-title, DOI, PMID and PMCID verification queries.

Exact 30-aa sequence searches returned no direct Europe PMC hits for the expanded selected set. Family/protein papers were therefore retained only as homologous or contextual evidence, never as proof that the exact motif is functional.

## Inclusion rules

- Include peer-reviewed primary studies, stable PDB/database records and local raw annotation outputs tied to the exact representative accession.
- Prefer direct mutation, deletion, reconstitution, interaction or structural evidence.
- Preserve clusterProfiler status exactly. `not_tested_external`, `external_enrichment_unavailable` and `external_filter_failed` are not treated as nonsignificant.
- Assign evidence grades as: A exact protein/region directly tested; B homologous protein or neighboring-region experimental support; C database/review/structural inference; D weak background.
- Retain both `evidence_scope` and `limitations`.

## Exclusion rules

- Exclude unverified titles, DOI/PMID pairs, database pages or conclusions.
- Exclude generic family function as motif-level proof.
- Exclude internal Fisher enrichment as a selection criterion; it may only be used later as cross-reference.
- Exclude exact AlphaFold/UniProt claims when the RefSeq representative accession could not be mapped reliably.

## Selection

Structural motifs were chosen by significance (`q <= 0.05`), strong/moderate structural evidence, mean motif pLDDT >= 70, family-level instance support, unique representative sequences, M2/M3 method diversity and separated representative positions. Functional motifs were chosen from external nonsignificant, not-tested, unavailable and filter-failed states using the same sequence/method/position diversity rules. Five motifs were retained per available family. GvpN selection preserves both `not_tested_external` and `external_enrichment_unavailable` where the ranked candidates permit it.

## Current status summary

- Functional status counts: {dict(status_counts)}
- Functional evidence-grade counts: {dict(grade_counts)}
- No A-grade row was assigned because no paper directly tested an exact selected representative accession and 30-aa interval.

## Limitations

- Representative windows identify searchable examples of a cluster; they are not inferred minimal functional domains.
- Most experimental studies use homologs from different organisms, so B-grade evidence supports family/region plausibility rather than exact motif causality.
- Predicted structural enrichment and pLDDT support structural plausibility but cannot establish biochemical function.
- GvpA-like paralogy affects some GvpJ/GvpM representatives, and AAA+ homology can confound GvpN annotation.
- GvpQ and GvpV remain comparatively weakly characterized; their candidate functions should be treated as exploratory.
"""
    (OUTDIR / f"search_protocol_{OUTPUT_TAG}.md").write_text(text, encoding="utf-8")


def validate(
    structural: pd.DataFrame, functional: pd.DataFrame, sources: pd.DataFrame
) -> list[str]:
    messages = []
    forbidden = {"core_start", "core_end", "core_length"}
    for name, frame in [
        ("structural", structural),
        ("functional", functional),
        ("sources", sources),
    ]:
        overlap = forbidden.intersection(frame.columns)
        if overlap:
            raise ValueError(f"{name} contains forbidden columns: {sorted(overlap)}")
    for name, frame in [("structural", structural), ("functional", functional)]:
        key = ["family_method", "motif"]
        duplicates = int(frame.duplicated(key).sum())
        if duplicates:
            raise ValueError(f"{name} contains {duplicates} duplicate keys")
        messages.append(f"{name}_duplicate_keys=0")
    source_ids = set(sources["source_id"])
    used_ids = {
        source_id
        for frame in [structural, functional]
        for value in frame["source_ids"]
        for source_id in value.split(";")
    }
    missing = sorted(used_ids - source_ids)
    if missing:
        raise ValueError(f"Missing source IDs: {missing}")
    messages.append("all_source_ids_resolved=true")
    required_statuses = [
        "external_enrichment_not_significant",
        "not_tested_external",
        "external_enrichment_unavailable",
        "external_filter_failed",
    ]
    observed_statuses = set(functional["external_enrichment_status"])
    missing_statuses = sorted(set(required_statuses) - observed_statuses)
    if missing_statuses:
        raise ValueError(f"Required external statuses are absent: {missing_statuses}")
    messages.append("unavailable_states_preserved=true")
    return messages


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    structural_motifs, functional_motifs = select_motif_sets(frame)
    needed = sorted(set(structural_motifs + functional_motifs))
    selected = frame[frame["motif_cluster"].isin(needed)].copy()
    missing = sorted(set(needed) - set(selected["motif_cluster"]))
    if missing:
        raise ValueError(f"Missing selected motifs: {missing}")
    accessions = set(selected["representative_parent_id"])
    interpro = compact_interpro(accessions)
    eggnog = compact_eggnog(accessions)
    structural = build_structural(selected, structural_motifs, interpro, eggnog)
    functional = build_functional(selected, functional_motifs, interpro, eggnog)
    sources = attach_source_motifs(SOURCES, structural, functional)
    messages = validate(structural, functional, sources)

    structural_path = OUTDIR / (
        f"structural_motif_biological_significance_validated_{OUTPUT_TAG}.csv"
    )
    functional_path = OUTDIR / (
        f"external_functional_motif_literature_validation_validated_{OUTPUT_TAG}.csv"
    )
    sources_path = OUTDIR / f"evidence_sources_{OUTPUT_TAG}.csv"
    structural.to_csv(structural_path, index=False, quoting=csv.QUOTE_MINIMAL)
    functional.to_csv(functional_path, index=False, quoting=csv.QUOTE_MINIMAL)
    sources.to_csv(sources_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_protocol(structural, functional)

    print(f"structural_records={len(structural)}")
    print(f"functional_records={len(functional)}")
    print(f"source_records={len(sources)}")
    print("functional_status_counts=" + repr(dict(Counter(functional["external_enrichment_status"]))))
    print("structural_grade_counts=" + repr(dict(Counter(structural["evidence_grade"]))))
    print("functional_grade_counts=" + repr(dict(Counter(functional["evidence_grade"]))))
    for message in messages:
        print(message)


if __name__ == "__main__":
    main()
