Magnet array optimization problems solved in python.
adv_hallbach.py contains code that will optimize magnetization directions (up, down, left, right) of magnets in a magnet array arranged in a line to produce the largest possible magnetic field at several sensor locations. 
The magnetization direction is determined by a direction value which translated as follows: +-1 as pointing right/left, +-2 as pointing front/back, +- 3 as pointing up/down. 
This creates a rather hard problem when using floating point genetic algorithm because typically, the direction will change slowly. It will be very hard to flip a direction especially in the z-direction (+3 to -3, it needs to turn front, right, left, back, and only then down)  

"Choi formulation" is based on "Optimization of Magnetization Directions in a 3-D Magnetic Structure" (https://ieeexplore.ieee.org/document/5467587) where a similar problem to the standard Halbach array optimization above is created in 3D.
Instead of one value representing all 3 directions, 3 values are used, one for each direction. This makes it easier to change the direction as each axis is controlled separately. 
There will be 3 states for each direction: +1, 0, or -1. In y direction, it will point front, neutral, and back, respectively.
In this problem, the magnets can also be stacked as several layers in x, y, and z directions creating a 3D array. Magnetization direction in each individual magnet is still up, down, left, and right.
hallbach_magnetization_formulation.py uses a 5 x 5 x 3 array, while hallbach_choiformulation.py uses 5 x 1 x 1.

"Inverse problem" is a problem where we have prescribed a target magnet field at several sensor locations. The magnet array needs to be configured to achieve the intended strength. 
The intended use is to see if the optimizer can generate a Halbach cylinder. However, the current version stopped short of that goal and simply prescribe a magnet field above the 3D halbach array used in the previous problems.
