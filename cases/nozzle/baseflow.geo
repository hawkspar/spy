// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
R=1;
L=60*R; H=15*R; o=1;
r=5e-4; h = 2.5e-4;

// Base 'rectangle'
Point(1) = {0, 0, 0, 100*r};
Point(2) = {L, 0, 0, 200*r};
Point(3) = {L, H, 0, 800*r};
Point(4) = {0, H, 0, 800*r};
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
Point(14) = {.9*R,      R,   0,    r};
Point(15) = {.9*R,      R+h, 0,    r};
Point(16) = {.9*R,  .95*R,   0,  7*r};
Point(17) = { 2*R,  .95*R,   0, 10*r};
Point(18) = { 2*R,  1.1*R,   0, 15*r};
Point(19) = {.9*R, 1.05*R,   0, 20*r};
// Less refined area 2
Point(20) = {.85*R,     R,   0,     r};
Point(21) = {.85*R,     R+h, 0,     r};
Point(22) = {.85*R, .85*R,   0,  10*r};
Point(23) = { 25*R,  .3*R,   0, 200*r};
Point(24) = { 25*R, 4.5*R,   0, 200*r};
Point(25) = {.85*R, 1.2*R,   0,  50*r};

// Main rectangle
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 6};
Line(5) = {5, 1};
// Nozzle
Line(6)  = {6,  21};
Line(7)  = {21, 15};
Line(8)  = {15,  9};
Line(9)  = {9,   7};
Line(10) = {7,   8};
Line(11) = {8,  14};
Line(12) = {14, 20};
Line(13) = {20,  5};
// Outer zone
Line(14) = {21, 25};
Line(15) = {25, 24};
Line(16) = {24, 23};
Line(17) = {23, 22};
Line(18) = {22, 20};
// Middle zone
Line(19) = {15, 19};
Line(20) = {19, 18};
Line(21) = {18, 17};
Line(22) = {17, 16};
Line(23) = {16, 14};
// Inner zone
Line(24) = {9,  13};
Line(25) = {13, 12};
Line(26) = {12, 11};
Line(27) = {11, 10};
Line(28) = {10,  8};

// Surfaces
Line Loop(1) = {1,  2,  3,  4,  6,  14, 15, 16, 17, 18, 13,  5};
Line Loop(2) = {14, 15, 16, 17, 18,-12,-23,-22,-21,-20,-19, -7};
Line Loop(3) = {19, 20, 21, 22, 23,-11,-28,-27,-26,-25,-24, -8};
Line Loop(4) = {24, 25, 26, 27, 28,-10, -9};
For i In {1:4}
	Plane Surface(i) = {i};
EndFor
