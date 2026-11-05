import contextlib
import io
import unittest
from unittest import mock
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package import cli
from package import prepare


class TestStandaloneLayout(unittest.TestCase):
    def test_package_imports_from_standalone_folder(self):
        module_path = Path(prepare.__file__).resolve()

        self.assertIn("src", module_path.parts)
        self.assertIn("exp5_external_functional_enrichment", module_path.parts)
        self.assertIn("package", module_path.parts)

    def test_defaults_read_stable_exp3_and_write_project_results(self):
        project_root = cli._default_project_root()

        self.assertEqual(project_root, ROOT.parents[3])
        self.assertEqual(cli._default_ncbi_root(), project_root / "data" / "ncbi")
        self.assertEqual(
            cli._default_exp3_dir(),
            project_root / "work" / "results" / "exp3_domain_discovery",
        )
        self.assertEqual(
            cli._default_output_dir(),
            project_root / "work" / "results" / "exp5_external_functional_enrichment",
        )
        self.assertEqual(cli._default_output_dir().name, "exp5_external_functional_enrichment")

    def test_motif_commands_only_accept_exp3_dir(self):
        exp3_dir = Path("results") / "exp3_domain_discovery"
        for command in ("make-motif-sets", "motif-windows"):
            args = cli.build_arg_parser().parse_args(
                [command, "--exp3-dir", str(exp3_dir)]
            )
            self.assertEqual(args.exp3_dir, exp3_dir)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.build_arg_parser().parse_args(
                        [command, "--exp2-dir", str(exp3_dir)]
                    )

    def test_output_root_can_be_overridden_for_server_runs(self):
        custom = str(ROOT / "server-results")
        with mock.patch.dict("os.environ", {"GVP_OUTPUT_DIR": custom}):
            self.assertEqual(cli._default_results_dir(), Path(custom).resolve())
            self.assertEqual(
                cli._default_exp3_dir(),
                Path(custom).resolve() / "exp3_domain_discovery",
            )



if __name__ == "__main__":
    unittest.main()
