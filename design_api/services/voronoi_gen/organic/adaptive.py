from typing import Tuple, List, Callable

def generate_adaptive_grid(
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    max_depth: int = 4,
    threshold: float = 0.1,
    error_metric: Callable[['OctreeNode'], float] = None
) -> 'OctreeNode':
    """
    Create an adaptive octree grid by recursively subdividing based on error_metric.
    """
    if error_metric is None:
        raise ValueError("error_metric must be provided")
    root = OctreeNode(bbox_min, bbox_max)
    root.subdivide(error_metric, max_depth, threshold)
    return root

# --- Adaptive Octree Grid for Level-of-Detail Sampling ---
class OctreeNode:
    """
    Represents an axis-aligned box region that can be recursively subdivided
    based on an error metric.
    """
    def __init__(self, bbox_min: Tuple[float,float,float], bbox_max: Tuple[float,float,float], depth: int = 0):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.depth = depth
        self.children: List['OctreeNode'] = []

    def subdivide(self,
                  error_metric: Callable[['OctreeNode'], float],
                  max_depth: int,
                  threshold: float):
        """
        Recursively subdivide this node if error_metric(node) > threshold
        and depth < max_depth.
        """
        err = error_metric(self)
        if self.depth >= max_depth or err <= threshold:
            return
        # compute midpoints
        x0,y0,z0 = self.bbox_min
        x1,y1,z1 = self.bbox_max
        mx, my, mz = (0.5*(x0+x1), 0.5*(y0+y1), 0.5*(z0+z1))
        boxes = [
            ((x0,y0,z0), (mx,my,mz)),
            ((mx,y0,z0), (x1,my,mz)),
            ((x0,my,z0), (mx,y1,mz)),
            ((mx,my,z0), (x1,y1,mz)),
            ((x0,y0,mz), (mx,my,z1)),
            ((mx,y0,mz), (x1,my,z1)),
            ((x0,my,mz), (mx,y1,z1)),
            ((mx,my,mz), (x1,y1,z1)),
        ]
        for bmin, bmax in boxes:
            child = OctreeNode(bmin, bmax, depth=self.depth+1)
            child.subdivide(error_metric, max_depth, threshold)
            self.children.append(child)