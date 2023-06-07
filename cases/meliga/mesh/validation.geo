// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
x_max=70; r_max=10;
l=50;
x_lim=x_max+l; r_lim=r_max+l;
h_min=1/45;    h_max=5/3;

// First fine rectangle around the nozzle
Point(1) = {0, 	   0, 	  0, h_min};
Point(2) = {x_max, 0, 	  0, h_max/3};
Point(3) = {x_max, r_max, 0, h_max/3};
Point(4) = {0, 	   r_max, 0, h_max/3};
// Sponge regions
Point(5) = {x_lim, 0, 	  0, h_max};
Point(6) = {x_lim, r_lim, 0, h_max};
Point(7) = {0, 	   r_lim, 0, h_max};

// Fine Loop
Line(1)  = {1,2};
Line(2)  = {2,3};
Line(3)  = {3,4};
Line(4)  = {4,1};
// Sponge loop
Line(5)  = {2,5};
Line(6)  = {5,6};
Line(7)  = {6,7};
Line(8)  = {7,4};

// Surfaces
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8,-3,-2};

For i In {1:2}
	Plane Surface(i) = {i};
EndFor
