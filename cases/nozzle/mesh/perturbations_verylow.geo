// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
R=1; L=50*R; H=15*R;
r=1.2e-3; h=1e-4;

// First base rectangle around the nozzle
Point(1) = {0, 0, 0,   30*r};
Point(2) = {R, 0, 0,   20*r};
Point(3) = {R, H, 0, 1000*r};
Point(4) = {0, H, 0, 1000*r};
// Second (largest) base rectangle
Point(5) = {L, 0, 0, 250*r};
Point(6) = {L, H, 0, 750*r};
// Actual nozzle
Point(9)  = {0, R,   0, r};
Point(10) = {0, R+h, 0, r};
Point(11) = {R, R,   0, r};
// Most refined area (left)
Point(12) = {.95*R,      R,   0, r};
Point(13) = {.95*R,  .99*R,   0, r};
Point(14) = {	 R,  .99*R,   0, r};
Point(15) = {	 R, 1.01*R,   0, r};
Point(16) = {.95*R, 1.01*R,   0, r};
Point(17) = {.95*R,      R+h, 0, r};
// Most refined area (center)
Point(18) = {1.1*R,  .99*R,   0, r};
Point(19) = {1.1*R, 1.01*R,   0, r};
// Less refined area (left)
Point(20) = {.3*R,     R,   0,     r};
Point(21) = {.3*R,  .7*R,   0,  10*r};
Point(22) = {	R,  .7*R,   0,  10*r};
Point(23) = {	R, 1.1*R,   0,  10*r};
Point(24) = {.3*R, 1.1*R,   0,  10*r};
Point(25) = {.3*R,     R+h, 0,     r};
// Less refined area (center)
Point(26) = {30*R, .25*R, 0, 200*r};
Point(27) = {45*R,  14*R, 0, 300*r};

// Largest Loop
Line(1)  = {1,  2};
Line(2)  = {2,  5};
Line(6)  = {6,  3};
Line(7)  = {3,  4};
Line(8)  = {4, 10};
Line(9)  = {10,25};
Line(10) = {25,24};
Line(11) = {24,23};
Line(12) = {23,27};
Line(13) = {27,26};
Line(14) = {26,22};
Line(15) = {22,21};
Line(16) = {21,20};
Line(17) = {20, 9};
Line(18) = {9,  1};
// Smaller Loop
Line(19) = {20,12};
Line(20) = {12,13};
Line(21) = {13,14};
Line(22) = {14,18};
Line(23) = {18,19};
Line(24) = {19,15};
Line(25) = {15,16};
Line(26) = {16,17};
Line(27) = {17,25};
// Tiny Loop
Line(28) = {17,11};
Line(29) = {11,12};
// Verticals
Line(30) = {2, 22};
Line(31) = {22,14};
Line(32) = {14,11};
Line(33) = {11,15};
Line(34) = {15,23};
Line(35) = {23, 3};
Line(36) = {5, 6};

// Surfaces
Line Loop(1) =   {1, 30, 15, 16, 17, 18};
Line Loop(2) = {-15, 31,-21,-20,-19,-16};
Line Loop(3) =  {21, 32, 29, 20};
Line Loop(4) =   {2, 36,  6,-35, 12, 13, 14,-30};
Line Loop(5) = {-14,-13,-12,-34,-24,-23,-22,-31};
Line Loop(6) =  {22, 23, 24,-33,-32};
Line Loop(7) =  {28, 33, 25, 26};
Line Loop(8) = {-27,-26,-25, 34,-11,-10};
Line Loop(9) =   {9, 10, 11, 35,  7,  8};

For i In {1:9}
	Plane Surface(i) = {i};
EndFor
