// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
L=34; H=6; e=1e-12;

// x=0 square
Point(1) = {0,e,0,1};
Point(2) = {L,e,0,1};
Point(3) = {L,H,0,1};
Point(4) = {0,H,0,1};

// x=0 square
Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};

// Surfaces
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Transfinite Curve {1,2,3,4}=100;
Transfinite Surface {:};
Recombine Surface{:};
