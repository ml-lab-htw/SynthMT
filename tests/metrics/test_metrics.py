import sys
import os
import unittest
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from synth_mt.benchmark.metrics import (
    as_instance_stack,
    _compute_iou_matrix,
    _compute_iot_matrix,
    _compute_skiou_matrix,
    anchor_points_to_instance_masks,
    _get_matches,
    _get_instance_mask_length,
    _get_ordered_anchor_points_length,
    _get_length_distribution,
    _get_length_distribution_anchor_points,
    _compute_average_curvature_from_ordered_skeleton_coords,
    _instance_mask_to_ordered_skeleton_anchor_coords,
    _get_average_curvatures,
    calculate_segmentation_metrics,
    _compute_histogram_distributions,
    calculate_downstream_metrics,
)


class TestMetrics(unittest.TestCase):
    def test_as_instance_stack(self):
        # Test with labeled mask
        labeled_mask = np.array([[1, 1, 0], [0, 2, 2]])
        stack = as_instance_stack(labeled_mask)
        self.assertEqual(stack.shape, (2, 2, 3))
        self.assertTrue(np.all(stack[0] == (labeled_mask == 1)))
        self.assertTrue(np.all(stack[1] == (labeled_mask == 2)))

        # Test with empty mask
        empty_mask = np.zeros((10, 10))
        stack = as_instance_stack(empty_mask)
        self.assertEqual(stack.shape, (0, 10, 10))

        # Test with already a stack
        input_stack = np.random.rand(3, 10, 10) > 0.5
        stack = as_instance_stack(input_stack)
        self.assertTrue(np.all(stack == input_stack))

        # Test with wrong dimensions
        with self.assertRaises(ValueError):
            as_instance_stack(np.zeros((1, 2, 3, 4)))

    def test_compute_iou_matrix(self):
        gt_masks = np.array(
            [
                [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
            ],
            dtype=bool,
        )
        pred_masks = np.array(
            [
                [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 1]],
            ],
            dtype=bool,
        )

        iou_matrix = _compute_iou_matrix(gt_masks, pred_masks)
        self.assertAlmostEqual(iou_matrix[0, 0], 3 / 4)
        self.assertAlmostEqual(iou_matrix[0, 1], 0)
        self.assertAlmostEqual(iou_matrix[1, 0], 0)
        self.assertAlmostEqual(iou_matrix[1, 1], 2 / 4)

        # Test with no GT masks
        iou_matrix_no_gt = _compute_iou_matrix(np.zeros((0, 3, 3)), pred_masks)
        self.assertEqual(iou_matrix_no_gt.shape, (0, 2))

        # Test with no pred masks
        iou_matrix_no_pred = _compute_iou_matrix(gt_masks, np.zeros((0, 3, 3)))
        self.assertEqual(iou_matrix_no_pred.shape, (2, 0))

    def test_compute_iot_matrix(self):
        gt_masks = np.array(
            [
                [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
            ],
            dtype=bool,
        )
        pred_anchor_points = [
            np.array([[0, 0], [0, 1], [1, 0]]),
            np.array([[2, 1], [2, 2]]),
        ]

        iot_matrix = _compute_iot_matrix(gt_masks, pred_anchor_points)
        self.assertAlmostEqual(iot_matrix[0, 0], 3 / 3)
        self.assertAlmostEqual(iot_matrix[0, 1], 0)
        self.assertAlmostEqual(iot_matrix[1, 0], 0)
        self.assertAlmostEqual(iot_matrix[1, 1], 1.0)

        # Test with no GT masks
        iot_matrix_no_gt = _compute_iot_matrix(np.zeros((0, 3, 3)), pred_anchor_points)
        self.assertEqual(iot_matrix_no_gt.shape, (0, 2))

        # Test with no pred masks
        iot_matrix_no_pred = _compute_iot_matrix(gt_masks, [])
        self.assertEqual(iot_matrix_no_pred.shape, (2, 0))

    def test_compute_skiou_matrix(self):
        gt_masks = np.array(
            [
                [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            ],
            dtype=bool,
        )
        pred_masks = np.array(
            [
                [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=bool,
        )

        skiou_matrix = _compute_skiou_matrix(gt_masks, pred_instance_masks=pred_masks)
        self.assertAlmostEqual(skiou_matrix[0, 0], 2 * 2 / (3 + 2))

        pred_anchor_points = [np.array([[0, 1], [1, 1], [2, 1], [3, 1]])]
        skiou_matrix_anchor = _compute_skiou_matrix(
            gt_masks, pred_instance_anchor_points=pred_anchor_points, spline_s=0
        )
        # This is tricky to assert an exact value due to spline fitting and rasterization
        self.assertTrue(0.6 < skiou_matrix_anchor[0, 0] <= 1.0)

    def test_anchor_points_to_instance_masks(self):
        anchor_points = [
            np.array([[0, 0], [0, 1], [1, 1]]),
            np.array([[2, 2], [3, 3]]),
        ]
        masks = anchor_points_to_instance_masks(anchor_points, (4, 4), width=1)
        self.assertEqual(masks.shape, (2, 4, 4))
        # Check some pixels
        self.assertTrue(masks[0, 0, 0])
        self.assertTrue(masks[0, 1, 0])
        self.assertTrue(masks[0, 1, 1])
        self.assertFalse(masks[0, 2, 2])
        self.assertTrue(masks[1, 2, 2])
        self.assertTrue(masks[1, 3, 3])

        # Test width=0
        masks_w0 = anchor_points_to_instance_masks(anchor_points, (4, 4), width=0)
        self.assertEqual(masks_w0.sum(), 5)  # 3 points in first, 2 in second

        # Test empty
        masks_empty = anchor_points_to_instance_masks([], (4, 4))
        self.assertEqual(masks_empty.shape, (0, 4, 4))

    def test_get_matches(self):
        iou_matrix = np.array(
            [
                [0.8, 0.1, 0.2],
                [0.2, 0.9, 0.3],
                [0.0, 0.2, 0.7],
            ]
        )
        matches, unmatched_p, unmatched_g = _get_matches(iou_matrix, iou_threshold=0.5)
        self.assertEqual(len(matches), 3)
        self.assertEqual(len(unmatched_p), 0)
        self.assertEqual(len(unmatched_g), 0)
        self.assertIn((0, 0), matches)
        self.assertIn((1, 1), matches)
        self.assertIn((2, 2), matches)

        matches_high_thresh, unmatched_p_high, unmatched_g_high = _get_matches(
            iou_matrix, iou_threshold=0.85
        )
        self.assertEqual(len(matches_high_thresh), 1)
        self.assertEqual(len(unmatched_p_high), 2)
        self.assertEqual(len(unmatched_g_high), 2)
        self.assertIn((1, 1), matches_high_thresh)
        self.assertIn(0, unmatched_p_high)
        self.assertIn(2, unmatched_p_high)
        self.assertIn(0, unmatched_g_high)
        self.assertIn(2, unmatched_g_high)

    def test_length_functions(self):
        # Test mask length
        mask = np.zeros((10, 10), dtype=bool)
        mask[2, 2:8] = True
        length = _get_instance_mask_length(mask)
        self.assertEqual(length, 6)

        # Test ordered anchor points length
        points = np.array([[2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7]])
        length_points = _get_ordered_anchor_points_length(points)
        self.assertAlmostEqual(length_points, 5)

        # Test length distributions
        masks = np.array([mask, mask])
        dist = _get_length_distribution(masks)
        self.assertTrue(np.all(dist == [6, 6]))

        points_list = [points, points]
        dist_anchor_points = _get_length_distribution_anchor_points(points_list)
        self.assertTrue(np.allclose(dist_anchor_points, [5, 5]))

    def test_curvature_functions(self):
        # Straight line
        line_coords = np.array([[i, 5] for i in range(10)])
        avg_curv, _, _ = _compute_average_curvature_from_ordered_skeleton_coords(
            line_coords, spline_s=0
        )
        self.assertAlmostEqual(avg_curv, 0.0, places=5)

        # Circular arc
        radius = 20
        center = (25, 25)
        theta = np.linspace(0, np.pi, 50)
        arc_coords = np.array(
            [center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)]
        ).T
        avg_curv_arc, _, _ = _compute_average_curvature_from_ordered_skeleton_coords(
            arc_coords, spline_s=0
        )
        self.assertAlmostEqual(avg_curv_arc, 1 / radius, delta=1e-3)

        # Test the wrapper
        mask = np.zeros((50, 50), dtype=bool)
        rr, cc = np.array(np.round(arc_coords), dtype=int).T
        mask[rr, cc] = True
        avg_curvatures = _get_average_curvatures(np.expand_dims(mask, axis=0), spline_s=0)
        self.assertAlmostEqual(
            avg_curvatures[0], 1 / radius, delta=0.7
        )  # less precision due to rasterization

    def test_calculate_segmentation_metrics(self):
        gt_mask = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 0],
            ]
        )
        pred_mask = np.array(
            [
                [1, 1, 0, 3],
                [1, 0, 0, 3],
                [0, 0, 2, 2],
                [0, 0, 2, 0],
            ]
        )
        metrics, _ = calculate_segmentation_metrics(gt_mask, pred_mask, thresholds=[0.5])
        self.assertIn("AP@0.50", metrics)
        self.assertIn("F1@0.50", metrics)
        self.assertIn("SKIoU_mean", metrics)

        # Test with anchor points
        pred_anchor_points = [
            np.array([[0, 0], [0, 1], [1, 0]]),  # Corresponds to instance 1
            np.array([[2, 2], [2, 3], [3, 2]]),  # Corresponds to instance 2
            np.array([[0, 3], [1, 3], [1, 3]]),  # Corresponds to instance 3
        ]
        metrics_anchor, _ = calculate_segmentation_metrics(
            gt_mask, anchor_points_instance_masks=pred_anchor_points, thresholds=[0.5]
        )
        self.assertIn("AP@0.50", metrics_anchor)
        self.assertIn("F1@0.50", metrics_anchor)
        self.assertIn("SKIoU_mean", metrics_anchor)

    def test_compute_histogram_distributions(self):
        data1 = np.array([1, 2, 2, 3, 3, 3])
        data2 = np.array([3, 4, 4, 5, 5, 5])
        dist1, dist2, bins, _, _ = _compute_histogram_distributions(
            data1, data2, num_bins_suggestion=5
        )
        self.assertEqual(dist1.shape, (4,))
        self.assertEqual(dist2.shape, (4,))
        self.assertEqual(bins.shape, (5,))
        self.assertAlmostEqual(dist1.sum(), 1.0)
        self.assertAlmostEqual(dist2.sum(), 1.0)

        # Test with empty data
        self.assertIsNone(_compute_histogram_distributions(np.array([]), data2))

        # Test with identical data
        dist1_ident, dist2_ident, _, _, _ = _compute_histogram_distributions(
            data1, data1, num_bins_suggestion=5
        )
        self.assertTrue(np.allclose(dist1_ident, dist2_ident))

    def test_calculate_downstream_metrics(self):
        gt_mask = np.zeros((20, 20), dtype=np.uint8)
        gt_mask[5, 5:15] = 1  # length 10
        gt_mask[15, 2:8] = 2  # length 6

        pred_mask = np.zeros((20, 20), dtype=np.uint8)
        pred_mask[5, 5:14] = 1  # length 9
        pred_mask[15, 2:9] = 2  # length 7

        metrics = calculate_downstream_metrics(gt_mask, pred_mask)
        self.assertIn("Length_KL", metrics)
        self.assertIn("Curvature_KL", metrics)
        self.assertIn("Avg Count GT", metrics)
        self.assertIn("Avg Count Pred", metrics)
        self.assertIn("Count Abs Err", metrics)
        self.assertIn("Count Rel Err", metrics)
        self.assertEqual(metrics["Avg Count GT"], 2)
        self.assertEqual(metrics["Avg Count Pred"], 2)
        self.assertAlmostEqual(metrics["Count Abs Err"], 0)
        self.assertAlmostEqual(metrics["Count Rel Err"], 0)
        self.assertGreater(metrics["Length_KL"], 0)
        self.assertAlmostEqual(
            metrics["Curvature_KL"], 0, delta=30
        )  # Should be near zero for straight lines

        # Test with anchor points
        pred_anchor_points = [
            np.array([[5, i] for i in range(5, 14)]),
            np.array([[15, i] for i in range(2, 9)]),
        ]
        metrics_anchor = calculate_downstream_metrics(
            gt_mask, anchor_points_instance_masks=pred_anchor_points
        )
        self.assertEqual(metrics_anchor["Avg Count GT"], 2)
        self.assertEqual(metrics_anchor["Avg Count Pred"], 2)
        self.assertAlmostEqual(metrics_anchor["Count Abs Err"], 0)
        self.assertAlmostEqual(metrics_anchor["Count Rel Err"], 0)
        self.assertGreater(metrics_anchor["Length_KL"], 0)
        self.assertAlmostEqual(metrics_anchor["Curvature_KL"], 0, delta=30)


if __name__ == "__main__":
    unittest.main()
