# Author: Rex Woodfield

# imports libraries
import math
import random as ran
import turtle as tr
import numpy as np


def vecLength3(array):
    """
    Finds the magnitude of a numpy vector with three dimensions

    :param array: the array vector
    :return: the magnitude of the vector
    """
    return math.sqrt(array[0] ** 2 + array[1] ** 2 + array[2] ** 2)


# global lighting variables for sun direction and ambient and specular lighting
sundir = np.array([1.0, 1.0, 1.0])
normsun = sundir / vecLength3(sundir)
ambientlight = 0.15
specularstrength = 1000
specularpower = 128
# bounces = 0


class Shape:
    """
    Base class used to define a primitive 3D shape
    """
    def __init__(self, center, color):
        self.ctr = np.array(center)
        self.ctr[1] = -self.ctr[1]
        self.col = np.array(color)


class Sphere(Shape):
    """
    Class used to define a 3D sphere object
    """
    def __init__(self, center, size, color):
        super().__init__(center, color)
        self.rad = size


class Plane(Shape):
    """
    Class used to define a 3D plane object
    """
    def __init__(self, center, normal, color):
        super().__init__(center, color)
        self.norm = np.array(normal) / vecLength3(np.array(normal))


class AABB(Shape):
    """
    Class used to define a 3D axis-aligned bounding box object
    (cannot be rotated)
    """
    def __init__(self, center, size, color):
        super().__init__(center, color)
        self.size = np.array(size)


def trace():
    """
    The main function that uses a raytracing algorithm to create a 3D image
    """
    # sets up the window and initializes variables
    win = tr.Screen()
    win.title("Raytracer")
    width = win.canvwidth
    height = win.canvheight

    # creates a virtual camera with the following position, field of view, and lens aspect ratio
    cam = np.array([0.0, 0.0, 0.0])
    screendist = 1.74
    aspect = float(width) / float(height)

    # creates a turtle object that will just be used for drawing small lines the size of individual pixels
    # also turns off tracing and hides the turtle cursor
    tracer = tr.Turtle()
    tr.tracer(0, 0)
    tracer.penup()
    tracer.hideturtle()

    # creates a list of 3D shapes to render later
    shapes = []
    for i in range(10):
        position = [ran.uniform(-5, 5), -2, ran.uniform(8, 20)]
        size = ran.uniform(.4, 1.1)
        shapes.append(Sphere([position[0], position[1] + size * 2 + size, position[2]], size,
                             [ran.uniform(.2, 1), ran.uniform(.2, 1), ran.uniform(.2, 1)]))
        shapes.append(AABB([position[0], position[1] + size, position[2]], size * 2,
                           [ran.uniform(.2, 1), ran.uniform(.2, 1), ran.uniform(.2, 1)]))

    # creates the ground plane
    shapes.append(Plane([0, -2, 0], [0, 1, 0], [0.49, 0.56, 0.57]))

    # loops through the columns of the screen
    for x in range(-width, width):
        tracer.penup()
        tracer.goto(x, -height)
        tracer.pendown()
        xpercent = inverseLerp(x, -width, width) * 2 - 1

        # loops through the rows for each column on the screen
        for y in range(-height, height):
            # finds the direction to raytrace from
            ypercent = inverseLerp(y, -height, height) * 2 - 1
            leftbottom = np.array([-xpercent * aspect, -ypercent, screendist])
            startdir = leftbottom / vecLength3(leftbottom)

            # finds the closest 3D object by checking which objects are in front of the current ray
            closestdist = 1000000.0
            normdiraway = np.array([0, 0, 0])
            closest = -1
            # creates a variable to accumulate color coming in from the light, used for reflections in the future
            accumulated = np.array([0.05, 0.05, 0.1])
            # loops through the shapes list to check which is closest
            for s in range(len(shapes)):
                # for each shape check if it intersects the ray, then save that and save the normal vector off
                # of that shape
                if isinstance(shapes[s], Sphere):
                    intersect = raySphereIntersect(cam, startdir, shapes[s].ctr, shapes[s].rad)
                    if intersect != -1.0 and intersect < closestdist:
                        closestdist = intersect
                        closest = s
                        diraway = shapes[closest].ctr - (startdir * closestdist + cam)
                        normdiraway = diraway / vecLength3(diraway)
                elif isinstance(shapes[s], AABB):
                    intersect = rayAABBIntersect(cam, startdir, shapes[s].ctr, shapes[s].size)
                    if intersect != -1.0 and intersect < closestdist:
                        closestdist = intersect
                        closest = s
                        diraway = shapes[closest].ctr - (startdir * closestdist + cam)
                        normdiraway = diraway / vecLength3(diraway)
                else:
                    intersect = rayPlaneIntersect(cam, startdir, shapes[s].ctr, shapes[s].norm)
                    if intersect != -1.0 and intersect < closestdist:
                        closestdist = intersect
                        closest = s
                        normdiraway = shapes[s].norm

            # if a shape was found calculate the view direction and find the accumulated light color using the
            # lighting function
            if closestdist != 1000000.0:
                point = startdir * closestdist
                viewdir = startdir - cam
                viewdir /= vecLength3(viewdir)
                accumulated += lighting(normdiraway, point, viewdir, shapes, closest, shapes[closest].col)

            # set the rgb pen color of the turtle and clamp the color from 0 to 1
            tracer.pencolor(max(min(accumulated[0], 1), 0),
                            max(min(accumulated[1], 1), 0),
                            max(min(accumulated[2], 1), 0))
            # goto the next x y coordinate
            tracer.goto(x, y)
        # update the screen
        tr.update()
    # when finished keep the window open
    win.mainloop()


def raySphereIntersect(origin, direction, center, size):
    """
    Checks if a ray intersects an arbitrary 3D sphere

    :param origin: The origin point of the ray
    :param direction: The normalized direction of the ray
    :param center: The center of the sphere
    :param size: The radius of the sphere
    :return: -1.0 if it doesn't intersect, the distance to the sphere on said ray otherwise
    """
    a = direction.dot(direction)
    s0_r0 = origin - center
    b = 2.0 * direction.dot(s0_r0)
    c = s0_r0.dot(s0_r0) - (size * size)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return -1.0
    else:
        return (-b - math.sqrt(disc)) / (2.0 * a)


def rayPlaneIntersect(origin, direction, center, normal):
    """
    Checks if a ray intersects an arbitrary 3D plane

    :param origin: The origin of the ray
    :param direction: The normalized direction of the ray
    :param center: The center of the plane
    :param normal: The normal vector facing away from the plane
    :return: -1.0 if it doesn't intersect, the distance to the plane on said ray otherwise
    """
    denominator = normal.dot(direction)
    if abs(denominator) > 0.0001:
        t = (center - origin).dot(normal) / denominator
        if t >= 0.0:
            return t
    return -1.0


def rayAABBIntersect(origin, direction, center, size):
    """
    Calculates the intersection point of a ray and an axis-aligned bounding box adapted from:

    https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms

    :param origin: The origin point of the ray
    :param direction: The normalized direction of the ray
    :param center: The center of the axis-aligned bounding box
    :param size: The length, width, and height of the axis-aligned bounding box
    :return: -1.0 if it doesn't intersect, the distance to the AABB on said ray otherwise
    """
    dir_fraction = np.empty(3, dtype=float)
    if direction[0] != 0:
        dir_fraction[0] = 1.0 / direction[0]
    if direction[1] != 0:
        dir_fraction[1] = 1.0 / direction[1]
    if direction[2] != 0:
        dir_fraction[2] = 1.0 / direction[2]

    lb = center + np.array([-size / 2, -size / 2, -size / 2])
    rt = center + np.array([size / 2, size / 2, size / 2])

    t1 = (lb[0] - origin[0]) * dir_fraction[0]
    t2 = (rt[0] - origin[0]) * dir_fraction[0]
    t3 = (lb[1] - origin[1]) * dir_fraction[1]
    t4 = (rt[1] - origin[1]) * dir_fraction[1]
    t5 = (lb[2] - origin[2]) * dir_fraction[2]
    t6 = (rt[2] - origin[2]) * dir_fraction[2]

    # the infinities are used here to eliminate NaNs that
    # are generated when the ray sits on a boundary plane
    tmin = max(-np.inf, min(t1, t2), min(t3, t4), min(t5, t6))
    tmax = min(np.inf, max(t1, t2), max(t3, t4), max(t5, t6))

    # if tmax < 0, ray (line) is intersecting AABB
    # but the whole AABB is behind the ray start
    if tmax < 0:
        return -1.0

    # if tmin > tmax, ray doesn't intersect AABB
    if tmin > tmax:
        return -1.0

    # t is the distance from the ray point
    # to intersection

    t = min(x for x in [tmin, tmax] if x >= 0)
    return t


def lighting(diraway, hitpoint, viewdir, shapes, skip, color):
    """
    Ultimately calculates the color of the traced pixel by checking the color of the objects and finding if they are
    in shadow or not.  Also calculates the diffuse lighting on an object by using the normal vector and the sun
    direction.

    :param diraway: The normal vector away from the point where the object was hit
    :param hitpoint: The point the ray intersects the object hit
    :param viewdir: The direction from the camera to the hitpoint
    :param shapes: The list of shapes in the virtual 3D scene
    :param skip: The index of the object the ray intersected with in the shapes list
    :param color: The color of the object the ray hit
    :return: The lit color of the object that the ray hit
    """
    # set shadows on the object to false by default
    shadow = False
    # loops through the list of shapes to check if the hitpoint is in shadow
    for s in range(len(shapes)):
        # skips the object hit to avoid casting shadows on itself
        if s != skip:
            if isinstance(shapes[s], Sphere):
                intersect = raySphereIntersect(hitpoint, -normsun, shapes[s].ctr, shapes[s].rad)
                if intersect != -1.0 and intersect > 0:
                    shadow = True
                    break
            elif isinstance(shapes[s], AABB):
                intersect = rayAABBIntersect(hitpoint, -normsun, shapes[s].ctr, shapes[s].size)
                if intersect != -1.0 and intersect > 0:
                    shadow = True
                    break
            else:
                intersect = rayPlaneIntersect(hitpoint, -normsun, shapes[s].ctr, shapes[s].norm)
                if intersect != -1.0 and intersect < 0.0006:
                    shadow = True
                    break
    if shadow:
        # if the object is in shadow return the global ambient light amount multiplied by the object color
        return ambientlight * color
    else:
        # if the object is not in shadow return the diffuse and specular lighting on the object plus the ambient
        # light amount multiplied by the object's color
        diffuse = max(diraway.dot(normsun) + ambientlight, ambientlight)
        reflectdir = reflect(-normsun, viewdir)
        reflectdir /= vecLength3(reflectdir)
        spec = viewdir.dot(reflectdir)
        spec = max(spec, 0)
        if spec > 0:
            spec = spec ** specularpower * specularstrength
        return (diffuse + spec) * color


def reflect(vector, norm):
    """
    Reflects a numpy vector over a given normal vector

    :param vector: The vector to reflect
    :param norm: The normal vector to reflect over
    :return: The reflected vector
    """
    return norm * 2.0 * vector.dot(norm) - vector


def inverseLerp(x, a, b):
    """
    Linear interpolation but backwards, finding the 0 to 1 interpolator value instead of the interpolated value

    :param x: The previously interpolated value
    :param a: The minimum value the interpolated value could be
    :param b: The maximum value the interpolated value could be
    :return: The interpolator used to find the interpolated value
    """
    return (x - a) / (b - a)


if __name__ == "__main__":
    # runs the raytracing algorithm
    trace()


# code for calculating reflections in the future:
#
# for b in range(bounces):
#     skip = closest
#     closestdist = 1000000.0
#     closest = -1
#
#     for s in range(len(spheres)):
#         if s != skip:
#             intersect = raySphereIntersect(point, normdiraway, spheres[s].ctr, spheres[s].rad)
#             if intersect != -1.0 and intersect < closestdist:
#                 closestdist = intersect
#                 closest = s
#
#     if closestdist != 1000000.0:
#         diraway = spheres[closest].ctr - (normdiraway * closestdist + cam)
#         normdirawayprev = normdiraway
#         normdiraway = diraway / vecLength3(diraway)
#         point = startdir * closestdist + normdiraway / 100.0
#         accumulated += lighting(normdiraway, normdirawayprev) * spheres[closest].col * 0.3
#     else:
#         break
