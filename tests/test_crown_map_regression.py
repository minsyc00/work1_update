from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usv_swarm import (  # noqa: E402
    build_experiment_output_dir,
    load_fleet_profile_json,
    load_map_for_planner,
    resolve_default_fleet_profile_path,
    validate_fleet_profile_for_map,
)


MAP_NAMES = (
    "static_obstacle_map_10x10_rect_obstacle",
    "static_obstacle_map_15x15_rect_triangle_small",
    "static_obstacle_map_20x20_two_obstacles",
    "static_obstacle_map_50x50_simple",
    "static_obstacle_map_200x200_mixed_obstacles",
    "static_obstacle_map_200x200_mixed_obstacles_s_polygon",
    "static_obstacle_map_400x400_mixed_obstacles",
)


def _map_path(name: str) -> Path:
    return ROOT / "maps" / name / f"{name}.json"


class CrownBundledMapProfileTests(unittest.TestCase):
    def test_every_bundled_map_declares_a_valid_physical_fleet(self) -> None:
        for name in MAP_NAMES:
            with self.subTest(map=name):
                map_path = _map_path(name)
                profile_path = resolve_default_fleet_profile_path(map_path)
                self.assertIsNotNone(profile_path)
                raw = validate_fleet_profile_for_map(map_path, profile_path)
                fleet, profiles, profile_id = load_fleet_profile_json(profile_path)
                config, _ = load_map_for_planner(
                    map_path,
                    fleet,
                    agent_profiles=profiles,
                    fleet_profile_id=profile_id,
                )
                defaults = raw.get("planning_defaults", {})
                self.assertEqual(len(profiles), config.fleet.num_agents)
                map_data = json.loads(map_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    len(profiles),
                    int(map_data["notes"]["recommended_usv_count"]),
                )
                self.assertIsNotNone(config.vehicle_footprint)
                self.assertIn("return_to_start", defaults)
                self.assertEqual(defaults.get("engine"), "certified_lns")
                for profile in profiles.values():
                    self.assertGreater(profile.vehicle_length, 0.0)
                    self.assertGreater(profile.vehicle_width, 0.0)
                    self.assertGreater(profile.min_turn_radius, 0.0)
                    self.assertLess(profile.vehicle_width, profile.coverage_width)
                output_name = build_experiment_output_dir(map_path, config).name
                self.assertIn("_sensor", output_name)
                self.assertIn("_hull", output_name)
                self.assertNotIn("_footprint", output_name)

    def test_profile_map_binding_rejects_the_wrong_map(self) -> None:
        first = _map_path(MAP_NAMES[0])
        wrong = resolve_default_fleet_profile_path(_map_path(MAP_NAMES[1]))
        with self.assertRaisesRegex(ValueError, "map_id mismatch"):
            validate_fleet_profile_for_map(first, wrong)

    def test_profile_loader_never_uses_sensor_as_hull_fallback(self) -> None:
        payload = {
            "fleet_profile_id": "missing_physical_hull",
            "agents": [
                {
                    "agent_id": 0,
                    "initial_state": {"x": 1.0, "y": 1.0, "psi_deg": 0.0},
                    "coverage_footprint": {"length_lf": 4.0, "width_wf": 2.0},
                    "motion_constraints": {"min_turn_radius": 0.5},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "fleet.json"
            profile_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "never used as a hull fallback"):
                load_fleet_profile_json(profile_path)


@unittest.skipUnless(
    os.environ.get("CROWN_RUN_MAP_REGRESSION") == "1",
    "set CROWN_RUN_MAP_REGRESSION=1 for expensive end-to-end map runs",
)
class CrownBundledMapEndToEndTests(unittest.TestCase):
    def _assert_map(self, name: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples" / "run_crown_mcpp_experiment.py"),
                    "--map",
                    str(_map_path(name)),
                    "--outputs-root",
                    temporary,
                    "--no-render",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=1200.0,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        report = json.loads(completed.stdout[completed.stdout.find("{"):])
        self.assertGreaterEqual(
            float(report["coverage_fraction"]) + 1.0e-9,
            float(report["coverage_target"]),
        )
        self.assertTrue(report["continuous_conflict_validated"])
        self.assertGreater(float(report["peak_rss_mb"]), 0.0)

    def test_10x10(self) -> None:
        self._assert_map(MAP_NAMES[0])

    def test_15x15(self) -> None:
        self._assert_map(MAP_NAMES[1])

    def test_20x20(self) -> None:
        self._assert_map(MAP_NAMES[2])

    def test_50x50(self) -> None:
        self._assert_map(MAP_NAMES[3])

    def test_200x200_mixed(self) -> None:
        self._assert_map(MAP_NAMES[4])

    def test_200x200_s_polygon(self) -> None:
        self._assert_map(MAP_NAMES[5])

    def test_400x400(self) -> None:
        self._assert_map(MAP_NAMES[6])


if __name__ == "__main__":
    unittest.main()
