// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
R=1;
L=35*R; H=6*R;
r=1e-3; h = 2.5e-4;

// Base 'rectangle'
Point(1) = {0, 0, 0,  50*r};
Point(2) = {L, 0, 0, 150*r};
Point(3) = {L, H, 0, 500*r};
Point(4) = {0, H, 0, 500*r};
// Actual nozzle
Point(5) = {0, R,   0, r};
Point(6) = {0, R+h, 0, r};
Point(7) = {R, R,   0, r};
// Most refined area
Point(8)  = {.95*R,      R,   0, r};
Point(9)  = {.95*R,      R+h, 0, r};
Point(10) = {.95*R,  .99*R,   0, r};
Point(11) = {1.1*R,  .99*R,   0, r};
Point(12) = {1.1*R, 1.01*R,   0, r};
Point(13) = {.95*R, 1.01*R,   0, r};
// Less refined area
Point(14) = { .3*R,     R,   0,     r};
Point(15) = { .3*R,     R+h, 0,     r};
Point(16) = { .3*R,  .7*R,   0,   5*r};
Point(17) = { 30*R,  .2*R,   0, 150*r};
Point(18) = { 30*R, 1.2*R,   0, 150*r};
Point(19) = { .3*R, 1.1*R,   0,   5*r};

// Main rectangle
Line(1)  = {1,   2};
Line(2)  = {2,   3};
Line(3)  = {3,   4};
Line(4)  = {4,   6};
Line(5)  = {5,   1};
// Nozzle
Line(6)  = {6,  15};
Line(7)  = {15,  9};
Line(8)  = {9,   7};
Line(9)  = {7,   8};
Line(10) = {8,  14};
Line(11) = {14,  5};
// Outer zone
Line(12) = {15, 19};
Line(13) = {19, 18};
Line(14) = {18, 17};
Line(15) = {17, 16};
Line(16) = {16, 14};
// Inner zone
Line(17) = {9,  13};
Line(18) = {13, 12};
Line(19) = {12, 11};
Line(20) = {11, 10};
Line(21) = {10,  8};

// Surfaces
Line Loop(1) =  {1,  2,  3,  4,  6, 12, 13, 14, 15, 16, 11,  5};
Line Loop(2) = {12, 13, 14, 15, 16,-10,-21,-20,-19,-18,-17, -7};
Line Loop(3) = {17, 18, 19, 20, 21, -9, -8};
For i In {1:3}
	Plane Surface(i) = {i};
EndFor
