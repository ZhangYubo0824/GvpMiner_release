import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_parameter_workflow.commands import (
    CommandContext,
    build_confirmatory_command,
    build_grouped_exp3_command,
    build_stability_selection_command,
)
from stable_parameter_workflow.contracts import (
    REPRESENTATIVE_FAMILIES,
    ParameterPair,
    WorkUnit,
)


def _context(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("family,sequence\nGvpK,AAAA\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    return CommandContext(
        project_root=ROOT.parent,
        python_executable=Path(sys.executable),
        dataset_csv=dataset,
        residue_cache_dirs=(cache,),
        residue_cache_fallback_dir=tmp_path / "cache-fallback",
    )


def _unit(tmp_path, stage="A", method="M2", profile="pilot"):
    return WorkUnit(stage, ParameterPair(30, 5), "GvpK", method, profile,
                    tmp_path / "outputs")


def _after(argv, flag):
    return argv[argv.index(flag) + 1]




def test_grouped_exp3_command_covers_all_stage_a_families(tmp_path):
    command = build_grouped_exp3_command(
        _context(tmp_path),
        _unit(tmp_path),
        families=REPRESENTATIVE_FAMILIES,
        methods=("M2",),
    )
    family_start = command.argv.index("--families") + 1
    method_start = command.argv.index("--methods")
    later_flags = [
        index for index, value in enumerate(command.argv)
        if index > method_start and value.startswith("--residue-cache-")
    ]
    method_end = min(later_flags, default=len(command.argv))
    assert command.argv[family_start:method_start] == REPRESENTATIVE_FAMILIES
    assert command.argv[method_start + 1:method_end] == ("M2",)
    assert command.log_path.name == "exp3_pair.log"
    assert len(command.expected_outputs) == 2 * len(REPRESENTATIVE_FAMILIES)


def test_pilot_stability_command_has_uniform_mcs_and_fixed_min_samples(tmp_path):
    command = build_stability_selection_command(_context(tmp_path), _unit(tmp_path))
    text = " ".join(command.argv)
    assert "--mcs 10 20 30 40 60 80" in text
    assert "--min-samples 10" in text
    assert "--repeats 10" in text
    assert "--seeds 0" in text
    assert "--null-repeats 0" in text


def test_confirmatory_uses_one_selected_mcs_and_full_profile(tmp_path):
    unit = _unit(tmp_path, stage="C", method="M3", profile="confirmatory")
    command = build_confirmatory_command(_context(tmp_path), unit, selected_mcs=40)
    text = " ".join(command.argv)
    assert "--mcs 40" in text
    assert "--mcs 10" not in text
    assert "--repeats 50" in text
    assert "--seeds 0 1 2 3 4" in text
    assert "--null-repeats 1000" in text
