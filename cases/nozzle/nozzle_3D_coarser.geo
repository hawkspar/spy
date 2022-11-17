// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
L=34; H=8.4;

// x=0 square
Point(1) = {0,-H/2,-H/2, 1};
Point(2) = {0,-H/2, H/2, 1};
Point(3) = {0, H/2,-H/2, 1};
Point(4) = {0, H/2, H/2, 1};

// x=0 square
Line(1)  = {1, 2};
Line(2)  = {2, 4};
Line(3)  = {4, 3};
Line(4)  = {3, 1};

// Surfaces
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};
out[]=Extrude {L,0,0} { Surface{1}; };

Transfinite Curve {:}=25;
Transfinite Surface {:};
Transfinite Volume {1};
Recombine Surface{:};
