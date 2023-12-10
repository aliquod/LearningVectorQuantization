from dataclasses import dataclass
import pandas as pd
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely
from copy import deepcopy

@dataclass
class LVQPrototype:
    prototype: pd.DataFrame  # a one-row dataframe that specifies the prototype; have the same columns as the training data (including class/category label)
    class_column: str  # name of the column that contains the response
    boundary_limits: tuple[float, float, float, float] | None = None # xmin, xmax, ymin, ymax
    
    @property
    def prototype_class(self):
        "Class/category of self"
        return self.prototype[self.class_column]
    
    @property
    def predictors(self):
        "Boolean mask of the predictor columns"
        return (self.prototype.index != self.class_column)
    
    def prototype_np(self, x_column, y_column):
        "Coordinates of the prototype as a numpy 2-by-1 array"
        return np.array(self.prototype[[x_column, y_column]])

    def prototype_shapely(self, x_column, y_column):
        "Same coordinates but as a shapely.Point"
        return shapely.Point(self.prototype_np(x_column, y_column))

    def __post_init__(self):
        # initialize a "path" dataframe that records the coordinates of the prototype in each iteration.
        self.path = self.prototype.transpose().copy()

        # initialize a large, square boundary (a shapely.Polygon object)
        if self.boundary_limits is None:
           self.boundary_limits = (-100, 100, -100, 100)
        xmin, xmax, ymin, ymax = self.boundary_limits
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.default_boundary = shapely.Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
        self.boundaries = {}
    
    
    def boundary_under(self, x_column, y_column):
        if (x_column, y_column) not in self.boundaries:
            self.boundaries[(x_column, y_column)] = deepcopy(self.default_boundary)
        return self.boundaries[(x_column, y_column)]

    def nudge(self, data_point, learning_rate):
        "Nudge the prototype towards/away from data_point"
        data_point = data_point.squeeze()
        data_class = data_point[self.class_column]

        step = learning_rate * (data_point[self.predictors] - self.prototype[self.predictors])
        if data_class == self.prototype_class:
            self.prototype.loc[self.predictors] += step
        else:
            self.prototype.loc[self.predictors] -= step

        # record the new coordinates in self.path
        self.path = pd.concat([self.path, self.prototype.to_frame().transpose()], axis=0)
    
    def distance(self, data_point):
        "Euclidean l2 distance between self.prototype and data_point"
        data_point = data_point.squeeze()
        return np.linalg.norm(data_point[self.predictors] - self.prototype[self.predictors])
    
    def plot(self, axis, x_column: str, y_column: str, color_mapper: dict, add_legend = False):
        "Plot the prototype as a large circle on the axis provided. Assumes the graph to be 2D."
        "color_mapper should be a dictionary that maps the category names to their colors on the graph."

        color = color_mapper[self.prototype_class]
        if add_legend:
            axis.scatter(self.prototype[x_column], self.prototype[y_column], s = 120, marker = "o",
                        facecolors="none", edgecolors = color, linewidths = 3, label = "prototype")
        else:
            axis.scatter(self.prototype[x_column], self.prototype[y_column], s = 120, marker = "o",
                        facecolors="none", edgecolors = color, linewidths = 3)

    def plot_path(self, axis, x_column: str, y_column: str, color_mapper, add_legend = False):
        "Plots the prototype as a circle, and its path as a thin line on the axis provided."
        color = color_mapper[self.prototype_class]
        self.plot(axis, x_column, y_column, color_mapper, add_legend)
        axis.plot(self.path[x_column], self.path[y_column], c = color, linewidth = 1, alpha = 0.6)
    
    def boundary_line_between(prototype1, prototype2, x_column: str, y_column: str) -> "Line2D":
        "Computes a Line2D object that bisects the segment between prototype1 and prototype2, in the xy-plane given by x_column and y_column."
        midpoint = (prototype1.prototype_np(x_column, y_column) + prototype2.prototype_np(x_column, y_column)) / 2
        direction = (prototype1.prototype_np(x_column, y_column) - prototype2.prototype_np(x_column, y_column))
        connecting_line = Line2D(direction, midpoint)   # the line segment joining the two prototypes
        bound_line = connecting_line.perpendicular_line()  # the line perpendicular to it that also passes through the midpoint, i.e. its bisector.
        
        return bound_line
    
    def constrict_decision_boundary(self, new_shapely_boundary: shapely.LineString, x_column, y_column):
        "Slices self.boundary off along new_boundary_line."
        current_boundary = self.boundary_under(x_column, y_column)
        assert current_boundary.contains(self.prototype_shapely(x_column, y_column))
        split_regions = shapely.ops.split(current_boundary, new_shapely_boundary).geoms
        new_region = [region for region in split_regions if region.contains(self.prototype_shapely(x_column, y_column))][0]
        self.boundaries[(x_column, y_column)] = new_region
    
    def constrict_boundaries_with(self, other_prototypes, x_column, y_column):
        "Computes the decision boundaries with a list of prototypes, then converts those boundaries into shapely.LineString to slice self.boundary."
        boundary_lines = [LVQPrototype.boundary_line_between(self, other, x_column, y_column) for other in other_prototypes]
        for line in boundary_lines:
            if line.slope is not None:
                line_left_end = (self.xmin, line.y_at_x(self.xmin))
                line_right_end = (self.xmax, line.y_at_x(self.xmax))
                shapely_line = shapely.LineString([line_left_end, line_right_end])
            else:
                x = line.point[0]
                shapely_line = shapely.LineString([(x, self.ymin), (x, self.ymax)])
                
            self.constrict_decision_boundary(shapely_line, x_column, y_column)

@dataclass
class LearningVectorQuantization:
    data: pd.DataFrame  # contains the training dataset (both predictors and the response)
    class_column: str   # name of the column containing the response (i.e. the class/category to be classified)
    n_prototypes: int    # number of prototypes for each class/category
    learning_rate: float = 0.01         # the learning rate by which prototypes nudge themselves towards/away from data points
    prototypes: list[LVQPrototype] | None = None      # a list of prototypes; give None to initialize.
    prototype_selector: str | None = "training"             # the method by which the prototypes are to be initialized.

    def __post_init__(self):
        self.boundaries = {}  # indexed by tuple of predictors, values are dictionaries containing shapely objects indexed by category names
        self.class_grouped_data = self.data.groupby(self.class_column)
        if self.prototypes is None:
            self.prototypes = []
            self.initialize_prototypes()
        
    def initialize_prototypes(self):
        match self.prototype_selector:
            case "training":
                for group_name, group_data in self.class_grouped_data:
                    # randomly sample self.n_prototypes points from each class/category, and use them as prototypes.
                    if group_data.shape[0] < self.n_prototypes:
                        raise ValueError(f"The group {group_name} only has {group_data.shape[0]} observations, impossible to sample {self.n_prototypes} from them.")
                    
                    class_prototypes_data = group_data.sample(n = self.n_prototypes)
                    for _, data_row in class_prototypes_data.iterrows():
                        new_prototype = LVQPrototype(prototype = data_row, class_column = self.class_column)
                        self.prototypes.append(new_prototype)
            
            case _:
                raise ValueError(f"'{self.prototype_selector}' is not a valid method of initializing prototypes.")
    
    def nearest_prototype(self, data_point):
        "Nearest in Euclidean distance."
        return min(self.prototypes, key = lambda prototype: prototype.distance(data_point))

    def prototypes_of_category(self, category: str):
        return [prototype for prototype in self.prototypes if prototype.prototype_class == category]

    def prototypes_except_category(self, category: str):
        return [prototype for prototype in self.prototypes if not prototype.prototype_class == category]

    def run_training_iteration(self):
        data_point = self.data.sample(n = 1)
        nearest_prototype = self.nearest_prototype(data_point)
        nearest_prototype.nudge(data_point, self.learning_rate)

    def train(self, n_iteration = 1000,
              learning_rate_updator = lambda lr: lr * 0.99, lr_threshold = 0.0001):
        
        start_time = perf_counter()
        for i in range(n_iteration):
            self.run_training_iteration()
            self.learning_rate = learning_rate_updator(self.learning_rate)
            if self.learning_rate < lr_threshold:
                break
        time_elapsed = perf_counter() - start_time
        time_per_iteration = time_elapsed / (i + 1)

        print(f"Training finished after {i+1} iterations and {time_elapsed:.3f}s ({time_per_iteration:.3f}s per iteration.)")

    def constrict_boundaries_of_category(self, category: str, x_column: str, y_column: str):
        "Asks every prototype of the category given to compute their decision boundaries."
        other_prototypes = self.prototypes_except_category(category)
        for prototype in self.prototypes_of_category(category):
            prototype.constrict_boundaries_with(other_prototypes, x_column, y_column)
    
    def constrict_all_boundaries(self, x_column: str, y_column: str):
        "Ask every prototype to compute their decision boundaries."
        for category, _ in self.class_grouped_data:
            self.constrict_boundaries_of_category(category, x_column, y_column)
    
    def boundary_of_category(self, category, x_column: str, y_column: str):
        "Takes the union of every prototype belonging to the category given."
        union_of_boundaries = shapely.unary_union([prototype.boundary_under(x_column, y_column) for prototype in self.prototypes_of_category(category)])
        return union_of_boundaries
    
    def compute_all_boundaries(self, x_column, y_column):
        new_boundaries = {}
        self.constrict_all_boundaries(x_column, y_column)
        for category, _ in self.class_grouped_data:
            new_boundaries[category] = self.boundary_of_category(category, x_column, y_column)
        self.boundaries[(x_column, y_column)] = new_boundaries
        
    def plot_shapely_boundary(self, boundary, axis, color, alpha: float = 0.2):
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        gpd_boundary = gpd.GeoSeries(boundary)
        gpd_boundary.plot(color = color, linewidth = 1, alpha = alpha, ax = axis)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
    
    def plot_all_boundaries(self, fig, axis, x_column: str, y_column: str, color_mapper: dict, alpha: float = 0.2):
        "Plot the decision boundaries (more precisely the regions they divide) as filled polygons on the axis given."
        if (x_column, y_column) not in self.boundaries:
            self.compute_all_boundaries(x_column, y_column)

        for category, _ in self.class_grouped_data:
            color = color_mapper[category]
            shapely_boundary = self.boundaries[(x_column, y_column)][category]
            self.plot_shapely_boundary(shapely_boundary, axis, color, alpha)

    def plot_training(self, fig, axis, x_column, y_column, colors_mapper, plot_paths: bool = False):
        "Plots the training data and the prototypes on the axis given."
        for category_name, category_data in self.class_grouped_data:
            plt.plot([],marker="", ls="", label = r"$\bf{" + category_name.capitalize() + "}$")   # place holder for column title
            axis.scatter(category_data[x_column], category_data[y_column],
                         c = colors_mapper[category_name], alpha=0.2,
                         label = "observation")
            if plot_paths:
                for i, prototype in enumerate(self.prototypes_of_category(category_name)):
                    prototype.plot_path(axis, x_column=x_column, y_column=y_column, color_mapper=colors_mapper, add_legend=(i==0))
            else:
                for i, prototype in enumerate(self.prototypes_of_category(category_name)):
                    prototype.plot(axis, x_column=x_column, y_column=y_column, color_mapper=colors_mapper, add_legend=(i==0))
    
    def plot(self, fig, ax, x_column, y_column, color_mapper, plot_paths: bool = False, v_adjust = -0.35):
        self.plot_training(fig, ax, x_column, y_column, color_mapper, plot_paths)
        self.plot_all_boundaries(fig, ax, x_column, y_column, color_mapper)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol = len(self.class_grouped_data), loc='upper center', bbox_to_anchor=(0.5, v_adjust))
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

@dataclass
class Line2D:
    direction: list[float] | np.ndarray
    point: float | list[float] | np.ndarray
    t_min: float = -10000
    t_max: float = 10000

    def __post_init__(self):
        self.direction = self.direction / np.linalg.norm(self.direction)
        if isinstance(self.point, float) or isinstance(self.point, int):
            self.point = np.array([0, self.point])
        elif isinstance(self.point, list):
            self.point = np.array(self.point)

    @property
    def intercept(self):
        return self.point[1] - self.point[0] * self.slope
    
    @property
    def slope(self):
        if not self.direction[0] == 0:
            return self.direction[1] / self.direction[0]
        return None
    
    def y_at_x(self, x):
        return self.slope * x + self.intercept
    
    def perpendicular_line(self, intersection: list[float] | np.ndarray | None = None):
        new_direction = np.matmul(np.array([[0,-1],[1,0]]), self.direction)
        if intersection is None:
            intersection = self.point
        return Line2D(new_direction, intersection)
    
    def plot(self, fig, axis, color = "blue"):
        axis.autoscale(False)
        if self.t_min >= self.t_max:
            return fig, axis
        limits = np.row_stack([self.point_at_time(self.t_min), self.point_at_time(self.t_max)])
        axis.plot(limits[:, 0], limits[:, 1], c = color)
        return fig, axis

    def plot_all(lines: list["Line2D"], fig, axis):
        for line in lines:
            line.plot(fig, axis)