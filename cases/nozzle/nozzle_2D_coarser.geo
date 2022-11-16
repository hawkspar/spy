// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
L=34; H=6;
r=1e3;

// Rectangle
Point(1) = {1, 0, 0, r};
Point(2) = {L, 0, 0, r};
Point(3) = {L, H, 0, r};
Point(4) = {1, H, 0, r};

// Main rectangle
Line(1)  = {1,   2};
Line(2)  = {2,   3};
Line(3)  = {3,   4};
Line(4)  = {4,   1};

Transfinite Line {1,3}=100;
Transfinite Line {2,4}=100;

// Surfaces
Line Loop(1) =  {1,  2,  3,  4};
Plane Surface(1) = {1};
Transfinite Surface {1} = {1};
Recombine Surface{1};
