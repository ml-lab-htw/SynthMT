import logging
from typing import Tuple, List, Union

import cv2
import numpy as np

from synth_mt.config.spots import SpotConfig

logger = logging.getLogger(__name__)


def _generate_polygon_vertices(
    center_y: int, center_x: int, avg_radius: int, min_verts: int, max_verts: int
) -> np.ndarray:
    """
    Helper to generate vertices for a random, irregular, convex-like polygon.

    This method creates irregularity in two ways:
    1.  The radius of each vertex is randomized.
    2.  The angle of each vertex is randomized.

    By sorting the random angles, we ensure the vertices are ordered
    sequentially around the center, which creates a simple (non-self-intersecting)
    polygon.

    Args:
        center_y: The y-coordinate of the polygon's center.
        center_x: The x-coordinate of the polygon's center.
        avg_radius: The average radius for the vertices.
        min_verts: The minimum number of vertices.
        max_verts: The maximum number of vertices.

    Returns:
        A NumPy array of [x, y] vertex coordinates, with dtype=np.int32.
    """
    # 1. Determine the number of vertices for this polygon.
    num_vertices = np.random.randint(min_verts, max_verts + 1)

    # 2. Generate random angles and sort them. This is the key to creating a
    #    simple, non-self-intersecting polygon. It ensures vertices are
    #    arranged "in order" around the center point.
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))

    # 3. Generate a random radius for each vertex to create irregularity.
    #    The radius is varied around the average radius provided.
    #    We also ensure the radius is at least 1 pixel.
    min_r = max(1, avg_radius * 0.7)
    max_r = avg_radius * 1.3
    radii = np.random.uniform(min_r, max_r, num_vertices)

    # 4. Convert from polar coordinates (angle, radius) to Cartesian (x, y).
    #    This is done in a vectorized way for performance.
    points_x = center_x + radii * np.cos(angles)
    points_y = center_y + radii * np.sin(angles)

    pos_noise_factor = 0.5
    x_pos_noise = np.random.uniform(
        -avg_radius * pos_noise_factor, avg_radius * pos_noise_factor, num_vertices
    )
    y_pos_noise = np.random.uniform(
        -avg_radius * pos_noise_factor, avg_radius * pos_noise_factor, num_vertices
    )
    points_x += x_pos_noise
    points_y += y_pos_noise

    # 5. Combine the x and y coordinates into a single (N, 2) array of vertices.
    #    np.vstack creates a (2, N) array, so we transpose it with .T
    points = np.vstack((points_x, points_y)).T

    # 6. Convert to the integer format required by OpenCV's drawing functions.
    return np.array(points, dtype=np.int32)


class SpotGenerator:
    """
    Manages the state and drawing of a collection of spots (fixed or moving).
    Now supports circular and polygonal shapes for fixed/moving spots.
    """

    def __init__(self, spot_cfg: SpotConfig, img_shape: Tuple[int, int]):
        """
        Initializes the generator and all its spot properties.

        Args:
            spot_cfg (SpotConfig): Configuration for the spots.
            img_shape (Tuple[int, int]): The (height, width) of the image.
        """
        logger.debug(
            f"Initializing SpotGenerator with {spot_cfg.count} spots, image shape {img_shape}."
        )
        self.cfg = spot_cfg
        self.img_shape = img_shape
        self.n_spots = self.cfg.count

        self._initialize_properties()
        logger.debug(f"SpotGenerator initialized. Total spots: {self.n_spots}.")

    def _initialize_properties(self):
        """Generates the initial state for all spot properties, including shapes."""
        logger.debug(f"Initializing properties for {self.n_spots} spots.")
        if self.n_spots == 0:
            self.coords, self.intensities, self.radii, self.kernel_sizes = [], [], [], []
            self.spot_shapes, self.polygon_vertices = [], {}
            logger.debug("No spots to initialize (n_spots is 0).")
            return

        h, w = self.img_shape
        self.coords = [
            (np.random.randint(0, h), np.random.randint(0, w)) for _ in range(self.n_spots)
        ]
        self.intensities = [
            np.random.uniform(self.cfg.intensity_min, self.cfg.intensity_max)
            for _ in range(self.n_spots)
        ]
        self.radii = [
            np.random.randint(self.cfg.radius_min, self.cfg.radius_max + 1)
            for _ in range(self.n_spots)
        ]
        self.kernel_sizes = [
            np.random.randint(self.cfg.kernel_size_min, self.cfg.kernel_size_max + 1)
            for _ in range(self.n_spots)
        ]

        # --- New: Determine shape and pre-calculate polygon vertices ---
        self.spot_shapes = []
        self.polygon_vertices = {}
        for i in range(self.n_spots):
            if np.random.random() < self.cfg.polygon_p:
                self.spot_shapes.append("polygon")
                (y, x) = self.coords[i]
                radius = self.radii[i]
                self.polygon_vertices[i] = _generate_polygon_vertices(
                    y,
                    x,
                    radius,
                    self.cfg.polygon_vertex_count_min,
                    self.cfg.polygon_vertex_count_max,
                )
                logger.debug(
                    f"Spot {i} initialized as a polygon with {len(self.polygon_vertices[i])} vertices."
                )
            else:
                self.spot_shapes.append("circle")
                logger.debug(f"Spot {i} initialized as a circle.")

        if self.n_spots > 0:
            logger.debug(
                f"Sample initial spot properties: Coords={self.coords[0]}, Shape={self.spot_shapes[0]}, Intensity={self.intensities[0]:.4f}, Radius={self.radii[0]}, Kernel Size={self.kernel_sizes[0]}."
            )
        logger.debug("Spot properties initialized.")

    def update(self):
        """
        Updates the state for the next frame. Only moves spots if max_step is defined.
        """
        logger.debug(f"Updating spot positions. Max step config: {self.cfg.max_step}.")
        if self.cfg.max_step is None or self.n_spots == 0:
            logger.debug("Spots are fixed or there are no spots. Skipping position update.")
            return

        h, w = self.img_shape
        step_x = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)
        step_y = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)

        new_coords = []
        for i, (y, x) in enumerate(self.coords):
            new_y = np.clip(y + step_y[i], 0, h - 1)
            new_x = np.clip(x + step_x[i], 0, w - 1)
            new_coords.append((new_y, new_x))
            # If it's a polygon, we need to move its vertices too
            if self.spot_shapes[i] == "polygon":
                self.polygon_vertices[i][:, 0] += step_x[i]
                self.polygon_vertices[i][:, 1] += step_y[i]

        self.coords = new_coords
        if self.n_spots > 0:
            logger.debug(
                f"Spots positions updated for {self.n_spots} spots. Sample new coord: {self.coords[0]}."
            )

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Draws the current spots onto the given image, supporting both circles and polygons."""
        logger.debug(f"Applying {self.n_spots} persistent spots to image of shape {img.shape}.")
        if self.n_spots == 0:
            logger.debug("No persistent spots to draw. Returning original image.")
            return img

        return SpotGenerator._draw_spots(
            img=img,
            spot_coords=self.coords,
            intensities=self.intensities,
            radii=self.radii,
            kernel_sizes=self.kernel_sizes,
            sigma=self.cfg.sigma,
            shapes=self.spot_shapes,
            polygon_vertices=self.polygon_vertices,
        )

    @staticmethod
    def _draw_spots(
        img: np.ndarray,
        spot_coords: List[Tuple[int, int]],
        intensities: Union[float, List[float]],
        radii: List[int],
        kernel_sizes: List[int],
        sigma: float,
        shapes: List[str],
        polygon_vertices: dict,
    ) -> np.ndarray:
        """
        Optimized: Draws all spots in batch (circles and polygons), then blurs and adds to image in one step.
        """
        if not spot_coords:
            return img

        if img.ndim == 3:
            h, w, _ = img.shape
            is_rgb = True
        else:
            h, w = img.shape
            is_rgb = False

        if isinstance(intensities, float):
            intensities = [intensities] * len(spot_coords)

        # Create a single mask for all spots
        mask = np.zeros((h, w), dtype=np.float32)
        circle_indices = [i for i, s in enumerate(shapes) if s == "circle"]
        polygon_indices = [i for i, s in enumerate(shapes) if s == "polygon"]

        # Draw all circles in batch
        for i in circle_indices:
            y, x = spot_coords[i]
            radius = radii[i]
            intensity = intensities[i]
            cv2.circle(mask, (int(x), int(y)), radius, intensity, -1)

        # Draw all polygons in batch
        if polygon_indices:
            polys = [polygon_vertices[i] for i in polygon_indices]
            poly_intensities = [intensities[i] for i in polygon_indices]
            for verts, intensity in zip(polys, poly_intensities):
                cv2.fillPoly(mask, [verts], intensity)

        # Use the largest kernel size for all spots (for speed)
        if kernel_sizes:
            kernel = 2 * max(kernel_sizes) + 1
        else:
            kernel = 3
        if kernel <= 0:
            kernel = 1
        elif kernel % 2 == 0:
            kernel += 1

        if sigma > 0 and kernel > 1:
            mask = cv2.GaussianBlur(mask, (kernel, kernel), sigma)

        if is_rgb:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        img += mask
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def apply_random_spots(img: np.ndarray, spot_cfg: SpotConfig) -> np.ndarray:
        """
        Adds stateless circular spots that are regenerated completely on every frame.
        """
        logger.debug(f"Applying {spot_cfg.count} random spots to image of shape {img.shape}.")
        n_spots = spot_cfg.count
        if n_spots == 0:
            logger.debug("No random spots to apply. Returning original image.")
            return img

        if img.ndim == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
        logger.debug(f"Image dimensions for random spots: H={h}, W={w}.")

        coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]
        intensities = [
            np.random.uniform(spot_cfg.intensity_min, spot_cfg.intensity_max)
            for _ in range(n_spots)
        ]
        radii = [
            np.random.randint(spot_cfg.radius_min, spot_cfg.radius_max + 1) for _ in range(n_spots)
        ]
        kernel_sizes = [
            np.random.randint(spot_cfg.kernel_size_min, spot_cfg.kernel_size_max + 1)
            for _ in range(n_spots)
        ]
        shapes = ["circle"] * n_spots

        if n_spots > 0:
            logger.debug(
                f"Sample random spot properties: Coords={coords[0]}, Intensity={intensities[0]:.4f}, Radius={radii[0]}, Kernel Size={kernel_sizes[0]}."
            )

        return SpotGenerator._draw_spots(
            img=img,
            spot_coords=coords,
            intensities=intensities,
            radii=radii,
            kernel_sizes=kernel_sizes,
            sigma=spot_cfg.sigma,
            shapes=shapes,
            polygon_vertices={},  # No polygons for random spots
        )
