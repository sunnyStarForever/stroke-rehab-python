import unittest
from types import SimpleNamespace

import numpy as np

from rehab_engine._stub import DepthSamplerConfig, PipelineConfig, SkeletonFilterConfig
from rehab_engine.sensor_pipeline import SensorPipeline
from rehab_engine.pose3d import (
    DepthSampleContext,
    DepthSampleMethod,
    DepthSampleResult,
    DepthSampler,
    EmaSkeletonFilter,
    Joint2D,
    Joint3D,
    JointProjector3D,
    REHAB22_NAMES,
    SkeletonSmoother,
)


def joints_at(x=50.0, y=50.0, score=0.9):
    return [
        Joint2D(name=name, x=x, y=y, score=score, raw_score=score, valid=True)
        for name in REHAB22_NAMES
    ]


class DepthSamplerTests(unittest.TestCase):
    def setUp(self):
        self.config = DepthSamplerConfig()
        self.sampler = DepthSampler(self.config)

    def test_context_foreground_sampling_and_background_rejection(self):
        depth = np.full((100, 100), 2000, dtype=np.uint16)
        depth[44:57, 44:57] = 1000
        joints = joints_at()
        context = self.sampler.estimate_context(depth, joints, 0.001)
        self.assertTrue(context.has_body_depth_ref)
        self.assertEqual(context.body_depth_ref_mm, 1000)
        self.assertTrue(context.has_background_depth_ref)
        self.assertEqual(context.background_depth_ref_mm, 2000)
        result = self.sampler.sample_joint(depth, joints[0], None, context, 0.001)
        self.assertTrue(result.valid)
        self.assertEqual(result.depth_raw_mm, 1000)
        self.assertEqual(result.method, DepthSampleMethod.FOREGROUND_WINDOW)

        background_joint = Joint2D(name="waist", x=80, y=80, score=0.9, valid=True)
        rejected = self.sampler.sample_joint(
            depth, background_joint, None, context, 0.001)
        self.assertFalse(rejected.valid)
        self.assertTrue(rejected.rejected_as_background)
        self.assertEqual(rejected.method, DepthSampleMethod.FAILED_BACKGROUND_HIT)

    def test_limb_inward_search_recovers_edge_joint(self):
        depth = np.zeros((80, 80), dtype=np.uint16)
        depth[45:56, 27:38] = 1000
        joint = Joint2D(name="left_wrist", x=20, y=50, score=0.9, valid=True)
        parent = Joint2D(name="left_elbow", x=50, y=50, score=0.9, valid=True)
        context = DepthSampleContext(body_depth_ref_mm=1000, has_body_depth_ref=True)
        result = self.sampler.sample_joint(depth, joint, parent, context, 0.001)
        self.assertTrue(result.valid)
        self.assertTrue(result.foreground_recovered)
        self.assertEqual(result.method, DepthSampleMethod.LIMB_INWARD_SEARCH)


class ProjectionAndFilterTests(unittest.TestCase):
    def test_projection_uses_rgb_intrinsics_and_preserves_debug_fields(self):
        projector = JointProjector3D()
        projector.set_intrinsics(100, 200, 50, 40)
        joints = joints_at(60, 50)
        samples = [
            DepthSampleResult(
                valid=True, depth_raw_mm=2000, depth_meters=2.0,
                method=DepthSampleMethod.FOREGROUND_WINDOW,
                reason="foreground_window")
            for _ in range(22)
        ]
        output = projector.project(joints, samples)
        self.assertAlmostEqual(output[0].x, 0.2)
        self.assertAlmostEqual(output[0].y, 0.1)
        self.assertEqual(output[0].z, 2.0)
        self.assertEqual(output[0].sampled_depth_mm, 2000)

    def test_ema_holds_invalid_and_uses_low_alpha_for_jump(self):
        config = SkeletonFilterConfig()
        ema = EmaSkeletonFilter(config)
        first = [Joint3D(x=0, y=0, z=1, score=1, valid=True) for _ in range(22)]
        initialized = ema.filter(first, 1 / 30)
        invalid = [Joint3D(valid=False) for _ in range(22)]
        held = ema.filter(invalid, 1 / 30)
        self.assertTrue(held[0].valid)
        self.assertEqual(held[0].z, 1)
        jumped = [Joint3D(x=0, y=0, z=2, score=1, valid=True) for _ in range(22)]
        filtered = ema.filter(jumped, 1 / 30)
        self.assertAlmostEqual(filtered[0].z, 1.2)

    def test_legacy_smoother_rejects_short_depth_jump(self):
        smoother = SkeletonSmoother(alpha=0.35)
        first = [Joint3D(z=1.0, score=1.0, valid=True) for _ in range(22)]
        output, stats = smoother.smooth(first, 1_000_000_000)
        self.assertEqual(stats.valid_output, 22)
        jump = [Joint3D(z=1.5, score=1.0, valid=True) for _ in range(22)]
        output, stats = smoother.smooth(jump, 1_033_000_000)
        self.assertEqual(stats.jump_rejected, 22)
        self.assertFalse(output[0].valid)

    def test_sensor_pipeline_lifts_2d_without_native_pose_objects(self):
        pipeline = SensorPipeline(PipelineConfig())
        pipeline._joint_projector.set_intrinsics(100, 100, 50, 50)
        depth = np.full((100, 100), 1000, dtype=np.uint16)
        points = [(55.0, 55.0, 0.9, True) for _ in range(22)]
        output = pipeline._lift_pose_to_3d(points, depth, None, 1_000_000_000)
        self.assertEqual(len(output), 22)
        self.assertTrue(all(point[4] for point in output))
        self.assertAlmostEqual(output[0][0], 0.05)
        self.assertAlmostEqual(output[0][1], 0.05)
        self.assertAlmostEqual(output[0][2], 1.0)

    def test_sensor_pipeline_respects_depth_100um_units(self):
        pipeline = SensorPipeline(PipelineConfig())
        pipeline._joint_projector.set_intrinsics(100, 100, 50, 50)
        depth = np.full((100, 100), 10000, dtype=np.uint16)
        points = [(50.0, 50.0, 0.9, True) for _ in range(22)]
        output = pipeline._lift_pose_to_3d(
            points, depth, None, 1_000_000_000, 0.0001)
        self.assertTrue(all(point[4] for point in output))
        self.assertAlmostEqual(output[0][2], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
