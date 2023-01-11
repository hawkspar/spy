// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
R=1;
L=35*R; H=6*R;
r=3e-2; h = 2.5e-4;

// Base 'rectangle'
Point(1) = {0, 0, 0,  2*r};
Point(2) = {L, 0, 0, 10*r};
Point(3) = {L, H, 0, 50*r};
Point(4) = {0, H, 0, 50*r};
// Actual nozzle
Point(5) = {0, R,   0, r};
Point(6) = {0, R+h, 0, r};
Point(7) = {R, R,   0, r};

// Lines
Line(1)  = {1,   2};
Line(2)  = {2,   3};
Line(3)  = {3,   4};
Line(4)  = {4,   6};
Line(5)  = {6,   7};
Line(6)  = {7,   5};
Line(7)  = {5,   1};

// Surface
Line Loop(1) =  {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};
