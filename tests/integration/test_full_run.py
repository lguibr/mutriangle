# File: tests/integration/test_full_run.py
"""
Integration test for full MuTriangle run.

NOTE: This test requires mutrimcts library to be installed.
It is skipped if mutrimcts is not available.
"""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mutriangle.cli import app
from mutriangle.config.app_config import APP_NAME, DATA_ROOT_DIR_NAME
from mutriangle.config_mgmt.storage import ensure_presets_saved

# Check if mutrimcts is available
try:
    import mutrimcts

    MUTRIMCTS_AVAILABLE = True
except ImportError:
    MUTRIMCTS_AVAILABLE = False

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def runner():
    """Provides a Typer CliRunner."""
    return CliRunner()


@pytest.mark.skipif(
    not MUTRIMCTS_AVAILABLE,
    reason="mutrimcts library not installed - required for full integration test",
)
@pytest.mark.integration
def test_toy_preset_run(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Tests running the 'mutriangle train toy' command.
    Uses a temporary directory to isolate the run data.
    Verifies directory structure created by Trieye.

    REQUIRES: mutrimcts>=1.0.0 to be installed
    NOTE: May fail in sandboxed environments due to Ray permission requirements
    """
    # Skip if Ray can't initialize (sandboxed environment)
    import ray

    try:
        if not ray.is_initialized():
            ray.init(
                num_cpus=1,
                ignore_reinit_error=True,
                _temp_dir=str(tmp_path / "ray_temp"),
            )
            ray.shutdown()
    except (PermissionError, OSError) as e:
        pytest.skip(f"Ray initialization failed in sandbox: {type(e).__name__}")

    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    print(f"\n--- Test Setup ---")
    print(f"Temp Path: {tmp_path}")
    print(f"Current CWD: {Path.cwd()}")

    # Ensure presets are saved
    ensure_presets_saved(set_default_if_missing=False)

    # Define expected directories
    data_root_parent = tmp_path / DATA_ROOT_DIR_NAME
    app_data_root = data_root_parent / APP_NAME
    config_dir = app_data_root / "configs"
    runs_dir = app_data_root / "runs"
    mlruns_dir = app_data_root / "mlruns"
    expected_toy_run_dir = runs_dir / "toy_run"

    # Set MLFLOW_TRACKING_URI
    expected_mlflow_uri = f"file:{mlruns_dir.resolve()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", expected_mlflow_uri)

    try:
        # Run the command
        print("\n--- Running mutriangle train toy ---")
        result = runner.invoke(app, ["train", "toy", "--log-level", "INFO"])

        print(f"Exit Code: {result.exit_code}")
        if result.exit_code != 0:
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)

        # Assertions
        assert result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.stderr}"

        # Check directories were created
        assert data_root_parent.is_dir(), f"Data root not created: {data_root_parent}"
        assert app_data_root.is_dir(), f"App data root not created: {app_data_root}"
        assert config_dir.is_dir(), f"Config dir not created: {config_dir}"
        assert (config_dir / "toy.config.json").is_file(), "Preset config missing"

        # Check run directory (Trieye structure)
        assert expected_toy_run_dir.is_dir(), (
            f"Run dir not found: {expected_toy_run_dir}"
        )
        assert mlruns_dir.is_dir(), f"MLflow dir not created: {mlruns_dir}"

        # Check subdirectories
        assert (expected_toy_run_dir / "checkpoints").is_dir(), (
            "Checkpoints dir missing"
        )
        assert (expected_toy_run_dir / "buffers").is_dir(), "Buffers dir missing"
        assert (expected_toy_run_dir / "logs").is_dir(), "Logs dir missing"
        assert (expected_toy_run_dir / "tensorboard").is_dir(), (
            "TensorBoard dir missing"
        )

        print("\n✅ Integration test passed!")

    finally:
        os.chdir(original_cwd)


def test_mutrimcts_import():
    """Test that mutrimcts can be imported (or skip gracefully)."""
    if MUTRIMCTS_AVAILABLE:
        import mutrimcts

        assert hasattr(mutrimcts, "SearchConfiguration")
        assert hasattr(mutrimcts, "run_mcts")
        print("✅ mutrimcts library is available")
    else:
        pytest.skip("mutrimcts library not installed")


def test_basic_cli_help(runner: CliRunner):
    """Test that CLI help works (doesn't require mutrimcts)."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "mutriangle" in result.stdout.lower() or "MuTriangle" in result.stdout
