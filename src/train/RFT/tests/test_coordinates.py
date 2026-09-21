from __future__ import annotations

import unittest

import numpy as np

from src.utils.coordinates import depth_boxes_to_habitat_world, detany_camera_to_habitat_world


class DetAnyCoordinateConversionTest(unittest.TestCase):
    def test_camera_axes_are_converted_to_habitat_axes(self) -> None:
        actual = detany_camera_to_habitat_world([[1.0, 2.0, 3.0]], np.eye(4))
        np.testing.assert_allclose(actual, [[1.0, -2.0, -3.0]])

    def test_sensor_translation_is_applied(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = [10.0, 20.0, 30.0]
        actual = detany_camera_to_habitat_world([[1.0, 2.0, 3.0]], pose)
        np.testing.assert_allclose(actual, [[11.0, 18.0, 27.0]])

    def test_sensor_rotation_is_applied(self) -> None:
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        actual = detany_camera_to_habitat_world([[1.0, 2.0, 3.0]], pose)
        np.testing.assert_allclose(actual, [[-3.0, -2.0, -1.0]])

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detany_camera_to_habitat_world([[1.0, 2.0]], np.eye(4))

    def test_depth_box_is_lifted_to_metric_habitat_world(self) -> None:
        centers, sizes = depth_boxes_to_habitat_world(
            [[0.0, 0.0, 4.0, 4.0]],
            np.full((4, 4), 2.0),
            np.eye(4),
            90.0,
        )
        np.testing.assert_allclose(centers, [[0.0, 0.0, -2.0]], atol=1e-6)
        np.testing.assert_allclose(sizes, [[0.05, 4.0, 4.0]], atol=1e-6)

    def test_depth_box_without_valid_depth_is_ignored(self) -> None:
        centers, sizes = depth_boxes_to_habitat_world(
            [[0.0, 0.0, 4.0, 4.0]],
            np.zeros((4, 4)),
            np.eye(4),
            90.0,
        )
        self.assertEqual(centers.shape, (0, 3))
        self.assertEqual(sizes.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
