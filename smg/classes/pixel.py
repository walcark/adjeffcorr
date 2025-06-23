from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Pixel:
    """ 
    Equivalent to a Quadratic Tree without 
    recursivity operations.
    """
    x0: float
    x1: float
    y0: float
    y1: float
    depth: int
    min_depth: int
    max_depth: int
    threshold: float

    def __post_init__(self):
        self.mx = 0.5 * (self.x0 + self.x1)
        self.my = 0.5 * (self.y0 + self.y1)

    @property
    def points(self) -> List[Tuple[float, float]]:
        """
        Returns the corners and the center of the pixel.
        """
        return [
            (self.x0, self.y0),
            (self.x1, self.y0),
            (self.x0, self.y1),
            (self.x1, self.y1),
            (self.mx, self.my)
        ]

    def check_subdivide(self, values: List[float]) -> bool:
        """
        Checks if the pixel needs a subdivision. If the maximum 
        difference between the evaluation points is over a given
        threshold, performs the subdivision.
        """
        self.values = values
        if (self.depth < self.min_depth):
            print(
                "Forced subdivision at depth=%d (min_depth=%d)", 
                self.depth, self.min_depth
            )
            return True
        max_diff = max(values) - min(values)
        return (self.depth < self.max_depth) and (max_diff > self.threshold)

    def get_subdivision(self) -> List["Pixel"]:
        """
        Returns the 4 sub-pixels of the current pixel instance.
        """
        args = dict(min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    threshold=self.threshold,
                    depth=self.depth + 1)
        return [
            Pixel(self.x0, self.mx, self.y0, self.my, **args),
            Pixel(self.mx, self.x1, self.y0, self.my, **args),
            Pixel(self.x0, self.mx, self.my, self.y1, **args),
            Pixel(self.mx, self.x1, self.my, self.y1, **args)
        ]