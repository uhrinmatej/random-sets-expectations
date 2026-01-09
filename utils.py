import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import math
from queue import PriorityQueue

#################################################
#### FUNCTIONS ##################################
#################################################

def polar_angle(x, y):
    """
    Calculate the counterclockwise angle from the positive x-axis to the vector (x,y).

    Parameters
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.

    Returns
    -------
    float
        The angle in radians (0 <= angle < 2*pi).
    """
    return math.atan2(y, x) % (2 * math.pi)


def rotation(x, y, angle):
    """
    Rotate a 2D point counterclockwise around the origin.

    Parameters
    ----------
    x : float
        The x-coordinate of the point to rotate.
    y : float
        The y-coordinate of the point to rotate.
    angle : float
        The rotation angle in radians. Positive values rotate counterclockwise.

    Returns
    -------
    tuple[float, float]
        A tuple containing the (x,y) coordinate of the rotated point.
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    arr = R @ np.array([x, y])
    return arr[0], arr[1]


def distance(first, second):
    """
    Calculate the Euclidean distance between two 2D points.

    Parameters
    ----------
    first : tuple[float, float]
        The first 2D point.
    second : tuple[float, float]
        The second 2D point.

    Returns
    -------
    float
        The Euclidean distance between the points.
    """
    return np.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


def dist_to_segment(pt, start, end):
    """
    Calculate the minimum Euclidean distance from a 2D point to a line segment.

    Parameters
    ----------
    pt : tuple[float, float]
        The (x,y) coordinates of the point.
    start : tuple[float, float]
        The (x,y) coordinates of the segment's start point.
    end : tuple[float, float]
        The (x,y) coordinates of the segment's end point.

    Returns
    -------
    float
        The shortest distance from the point to the line segment.
    """
    edge_vector = (end[0] - start[0], end[1] - start[1])
    pt_vector = (pt[0] - start[0], pt[1] - start[1])

    segment_length = edge_vector[0] ** 2 + edge_vector[1] ** 2

    if segment_length == 0:
        return distance(pt, start)

    t = max(0, min(1, (np.array(pt_vector) @ np.array(edge_vector)) / segment_length))

    projection = (start[0] + t * edge_vector[0], start[1] + t * edge_vector[1])

    return distance(pt, projection)


#################################################
#### CLASSES ####################################
#################################################

class ConvexPolygon:
    """
    A class representing a 2D convex polygon with various geometric operations.

    Parameters
    ----------
    points : list of [float, float]
        A list of [x, y] coordinate pairs defining the vertices of the polygon.
        The polygon should be convex and vertices should be given in counter-clockwise order.

    Attributes
    ----------
    points : numpy.ndarray
        Array of shape (n+1, 2) containing the vertices of the polygon with the first vertex repeated at the end.
        The order is determined by the polar angle of the edge vectors.
    n : int
        Number of vertices in the polygon.
    """

    def __init__(self, points):
        self.n = len(points)
        self.points = self._sort_points(points)

    def _sort_points(self, points):
        """
        Sort the vertices of the polygon to the correct order (determined by the polar angle of the edge vectors).

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (n, 2) containing the vertices of the polygon.

        Returns
        -------
        numpy.ndarray
            Array of shape (n+1, 2) with vertices sorted in the correct order
            and the first vertex repeated at the end.
        """
        points = np.array(points)
        if self.n==1: return points
        else:
            points = np.roll(points, -np.lexsort((points[:, 0], points[:, 1]))[0], axis=0)
            return np.concatenate([np.array(points), [points[0]]])

    def area(self):
        """
        Calculate the area of the convex polygon using the shoelace formula.

        Returns
        -------
        float
            The area of the polygon.
        """
        x = self.points[:-1, 0]
        y = self.points[:-1, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def is_in(self, x, y):
        """
        Check if a point lies inside the convex polygon.

        Parameters
        ----------
        x : float
            x-coordinate of the point.
        y : float
            y-coordinate of the point.

        Returns
        -------
        bool
            True if the point is inside or on the boundary of the polygon, False otherwise.
        """
        if self.n < 3:
            return False

        edge_vectors = self.points[1:] - self.points[:-1]
        point_vecs = np.array([x, y]) - self.points[:-1]
        cross_products = edge_vectors[:, 0] * point_vecs[:, 1] - edge_vectors[:, 1] * point_vecs[:, 0]
        return np.all(cross_products >= 0) or np.all(cross_products <= 0)


    def oriented_distance(self, x, y):
        """
        Calculate the oriented distance from a point to the polygon.

        The distance is positive if the point is outside the polygon,
        negative if inside, and zero if on the boundary.

        Parameters
        ----------
        x : float
            x-coordinate of the point.
        y : float
            y-coordinate of the point.

        Returns
        -------
        float
            The oriented distance from the point to the polygon.
        """
        dist_to_edges = [dist_to_segment((x,y), self.points[i], self.points[i+1]) for i in range(self.n)]
        return np.min(dist_to_edges) * (-1 if self.is_in(x, y) else 1)


    def plot(self, filled=False, alpha=1, color=None):
        """
        Plot the convex polygon using matplotlib.

        Parameters
        ----------
        filled : bool, optional
            If True, fills the polygon with the specified color.
            If False, only the boundary is drawn.
        alpha : float, optional
            Opacity of the polygon.
        color : str or tuple, optional
            Color of the polygon.

        Returns
        -------
        None
            Displays a plot.
        """
        if filled:
            plt.gca().add_patch(patch.Polygon(self.points, fc=color, alpha=alpha))
        else:
            plt.gca().plot(self.points[:,0], self.points[:,1], color=color, alpha=alpha)



class SimpleRandomConvexSet:
    """
    A class representing a (simple) random set composed of convex polygons with associated probabilities.

    This class provides methods to compute some statistical properties of random convex sets,
    including the Aumann expectation, coverage function, Vorobyev expectation (via simulation),
    and Oriented Distance Average (ODA) expectation (via simulation).

    Parameters
    ----------
    sets : list of ConvexPolygon
        A list of ConvexPolygon objects representing the possible realizations of the random set.
    probs : array-like, optional
        A list or array of probabilities corresponding to each convex polygon in `sets`.
        If None (default), a uniform distribution over all polygons is assumed.

    Attributes
    ----------
    k : int
        Number of convex polygons (realizations)in the random set.
    probs : numpy.ndarray
        Array of probabilities for each convex polygon.
    sets : list of ConvexPolygon
        The collection of convex polygons that make up the random set.
    _aumann : ConvexPolygon or None
        Cached Aumann expectation of the random set.
    """

    def __init__(self, sets, probs=None):
        self.k = len(sets)
        if probs is None:
            self.probs = np.ones(self.k) / self.k
        else:
            self.probs = np.array(probs)

        self.sets = sets

        self._aumann = None

    def aumann(self):
        """
        Compute the Aumann expectation of the random convex set. In this case, the Aumann expectation is the Minkowski average of the convex polygons,
        weighted by their probabilities.

        Returns
        -------
        ConvexPolygon
            A convex polygon representing the Aumann expectation of the random set.

        Notes
        -----
        The result is cached after the first computation for better performance.
        """
        if self._aumann is not None:
            return self._aumann
        else:
            output_polygon = []
            first = self.probs @ np.array([poly.points[0] for poly in self.sets])
            output_polygon.append(first)

            pq = PriorityQueue()
            for i,poly in enumerate(self.sets):
                for j in range(poly.n):
                    vec = self.probs[i] * (poly.points[j+1] - poly.points[j])
                    pq.put((polar_angle(vec[0], vec[1]), *vec))

            while not pq.empty():
                _, x, y = pq.get()
                output_polygon.append(output_polygon[-1] + np.array([x, y]))

            self._aumann = ConvexPolygon(output_polygon[:-1])
            return self._aumann


    def coverage_function(self, x, y):
        """
        Compute the coverage function of the random set at point [x,y].

        Parameters
        ----------
        x : float
            x-coordinate of the point.
        y : float
            y-coordinate of the point.

        Returns
        -------
        float
            The probability that the point [x,y] is contained in a random
            realization of the set.
        """
        cvg_prob = 0
        for poly,prob in zip(self.sets,self.probs):
            if poly.is_in(x,y): cvg_prob += prob
        return cvg_prob

    def vorobyev_sim(self, nx, ny, xlim, ylim):
        """
        Compute and visualize the Vorobyev expectation of the random set via simulation.

        The Vorobyev expectation is the level set of the coverage function that has
        the same expected area as the random set. This method computes it numerically
        by evaluating the coverage function on a grid and finding the appropriate level.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        xlim : tuple of float
            The (min, max) x-coordinates for the grid.
        ylim : tuple of float
            The (min, max) y-coordinates for the grid.

        Returns
        -------
        None
            Displays a plot with two subplots:
            - Left: The coverage function
            - Right: The Vorobyev expectation (binary mask)
        """

        x_space = np.linspace(*xlim, nx)
        y_space = np.linspace(*ylim, ny)
        cvg_probs = np.zeros((nx, ny))
        for i, x in enumerate(x_space):
            for j, y in enumerate(y_space):
                cvg_probs[i,j] = self.coverage_function(x,y)

        avg_area = self.probs @ np.array([poly.area() for poly in self.sets])
        level_space = np.linspace(0, 1, 1000)[1:]
        areas = np.zeros_like(level_space)
        for i,t in enumerate(level_space):
            areas[i] = np.count_nonzero(cvg_probs >= t) / (nx * ny) * (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])

        diffs = np.abs(areas-avg_area)
        level = level_space[np.max(np.where(diffs == diffs.min()))]


        fig, ax = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw={'width_ratios': [44, 2, 44]}, constrained_layout=True)
        a0 = ax[0].pcolormesh(x_space, y_space, cvg_probs.T, cmap="hot")
        ax[0].set_title("Coverage function")
        plt.colorbar(a0, cax=ax[1])
        ax[2].pcolormesh(x_space, y_space, cvg_probs.T >= level, cmap="hot")
        ax[2].set_title("Vorobyev expectation")
        plt.show()

    def oda_sim(self, nx, ny, xlim, ylim):
        """
        Compute and visualize the Oriented Distance Average (ODA) of the random set via simulation.

        The ODA is the set of points where the mean oriented distance function
        is non-positive. This method computes it numerically by evaluating the
        mean oriented distance function on a grid.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        xlim : tuple of float
            The (min, max) x-coordinates for the grid.
        ylim : tuple of float
            The (min, max) y-coordinates for the grid.

        Returns
        -------
        None
            Displays a plot with two subplots:
            - Left: The mean oriented distance function
            - Right: The ODA expectation (binary mask)
        """
        x_space = np.linspace(*xlim, nx)
        y_space = np.linspace(*ylim, ny)
        mean_oriented_dist = np.zeros((nx, ny))
        for i, x in enumerate(x_space):
            for j, y in enumerate(y_space):
                mean_oriented_dist[i, j] = self.probs @ np.array([poly.oriented_distance(x, y) for poly in self.sets])

        cmap_extremes = max(mean_oriented_dist.max(), -mean_oriented_dist.min())
        fig, ax = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw={'width_ratios': [44, 2, 44]}, constrained_layout=True)
        a0 = ax[0].pcolormesh(x_space, y_space, mean_oriented_dist.T, cmap="bwr", vmin=-cmap_extremes, vmax=cmap_extremes)
        ax[0].set_title("Mean oriented distance")
        ax[1] = plt.colorbar(a0, cax=ax[1])
        ax[2].pcolormesh(x_space, y_space, mean_oriented_dist.T <= 0, cmap="hot")
        ax[2].set_title("Oriented distance average")
        plt.show()



