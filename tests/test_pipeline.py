"""
tests/test_pipeline.py
Automated validation of predictive maintenance pipeline and inference engine.
"""

import unittest
from predict import PredictiveMaintenanceEngine

class TestPredictiveMaintenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PredictiveMaintenanceEngine()

    def test_healthy_machine(self):
        healthy_telemetry = {
            "Type": "L",
            "Air_Temperature_K": 298.15,
            "Process_Temperature_K": 308.65,
            "Rotational_Speed_RPM": 1500.0,
            "Torque_Nm": 40.0,
            "Tool_Wear_min": 15.0,
            "Vibration_mms": 1.8
        }
        res = self.engine.predict_single(healthy_telemetry)
        self.assertFalse(res["is_failure_predicted"])
        self.assertGreater(res["health_index"], 80.0)
        self.assertEqual(res["failure_mode"], "No Failure")
        self.assertGreater(res["rul_hours"], 100.0)
        self.assertEqual(res["recommendation"]["priority"], "LOW")

    def test_tool_wear_failure(self):
        worn_tool_telemetry = {
            "Type": "L",
            "Air_Temperature_K": 300.0,
            "Process_Temperature_K": 310.0,
            "Rotational_Speed_RPM": 1500.0,
            "Torque_Nm": 40.0,
            "Tool_Wear_min": 240.0,
            "Vibration_mms": 5.5
        }
        res = self.engine.predict_single(worn_tool_telemetry)
        self.assertTrue(res["is_failure_predicted"])
        self.assertLess(res["health_index"], 50.0)
        self.assertEqual(res["recommendation"]["priority"], "HIGH")

    def test_heat_dissipation_failure(self):
        hdf_telemetry = {
            "Type": "L",
            "Air_Temperature_K": 300.0,
            "Process_Temperature_K": 305.0,  # temp diff = 5.0 K (< 8.6 K)
            "Rotational_Speed_RPM": 1250.0,  # < 1380 rpm
            "Torque_Nm": 55.0,
            "Tool_Wear_min": 30.0,
            "Vibration_mms": 4.8
        }
        res = self.engine.predict_single(hdf_telemetry)
        self.assertTrue(res["is_failure_predicted"])
        self.assertIn(res["failure_mode"], ["HDF", "OSF", "PWF", "TWF"])

    def test_overstrain_failure(self):
        osf_telemetry = {
            "Type": "L",
            "Air_Temperature_K": 298.0,
            "Process_Temperature_K": 308.0,
            "Rotational_Speed_RPM": 1400.0,
            "Torque_Nm": 68.0,
            "Tool_Wear_min": 210.0, # Strain = 210 * 68 = 14280 > 11000
            "Vibration_mms": 5.8
        }
        res = self.engine.predict_single(osf_telemetry)
        self.assertTrue(res["is_failure_predicted"])
        self.assertLess(res["rul_hours"], 50.0)

if __name__ == "__main__":
    unittest.main()
