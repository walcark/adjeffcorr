############################################################################
# Loading useful modules                                                   #
############################################################################
from typing import List, Callable, Any, Tuple, Dict
from smg.quadtree.pixel import Pixel
import numpy as np
import logging
logger = logging.getLogger(__name__)


def run_quadtree_subdivision(
    root: Pixel,
    func: Callable[..., Any],
    **func_kwargs
) -> List[Pixel]:
    """
    Initializes and runs the full recursive quadtree subdivision.

    Args:
        root (Pixel): Root pixel to start from.
        func (Callable): Evaluation function for pixel coordinates.
        cache (dict): Cache of evaluated coordinates.
        **func_kwargs: Additional arguments for the evaluation function.

    Returns:
        List[Pixel]: Final list of non-subdivided pixels.
    """
    cache = {}
    return subdivide_recursively(
        to_process=[root],
        final=[],
        func=func,
        cache=cache,
        **func_kwargs
    )


def evaluate_pixels(
    pixels: List[Pixel], 
    func: Callable[..., List[float]],
    cache: Dict[Tuple[float, float], float],
    **kwargs
) -> List[bool]:
    """
    Evaluate unique pixel coordinates using a cached function, and
    determine if each pixel needs to be subdivided.
    """
    print(f"Evaluating {len(pixels)} pixels ...")

    unique_coords, remap_indices = [], []
    coord_index_map = {}

    for pixel in pixels:
        indices = []
        for coord in pixel.points:
            if coord not in coord_index_map:
                coord_index_map[coord] = len(unique_coords)
                unique_coords.append(coord)
            indices.append(coord_index_map[coord])
        remap_indices.append(indices)

    to_evaluate = [
        coord for coord in unique_coords 
        if coord not in cache
    ]
    print(f"→ {len(to_evaluate)} unique coordinates to evaluate ...")

    if to_evaluate:
        results = func(to_evaluate, **kwargs)
        if len(results) != len(to_evaluate):
            print("Function returned {} values for {} coords ...".format(
                len(results), len(to_evaluate))
            )
                  
        for coord, value in zip(to_evaluate, results):
            cache[coord] = value

    values = [
        tuple(cache[unique_coords[i]] for i in group) 
        for group in remap_indices
    ]
    return [pix.check_subdivide(v) for pix, v in zip(pixels, values)]


def subdivide_recursively(
    to_process: List[Pixel],
    final: List[Pixel],
    func: Callable[..., Any],
    cache: Dict[Tuple[float, float], float],
    **kwargs
) -> List[Pixel]:
    """
    Recursively subdivide pixels using evaluation function with caching.
    """
    should_subdivide = evaluate_pixels(to_process, func, cache, **kwargs)

    next_pixels = []
    for subdivide, pixel in zip(should_subdivide, to_process):
        if subdivide:
            next_pixels.extend(pixel.get_subdivision())
        else:
            final.append(pixel)

    if not next_pixels:
        print("No more subdivisions; reached final resolution.")
        return final

    return subdivide_recursively(next_pixels, final, func, cache, **kwargs)


def extract_pixel_centers(
    pixels: List[Pixel]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pixel center coordinates and their center values.
    """
    centers = np.array([[p.mx, p.my] for p in pixels])
    values = np.array([p.values[4] for p in pixels])
    return centers, values
