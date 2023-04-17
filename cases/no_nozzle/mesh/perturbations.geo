// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
R=1;
L=50*R; H=10*R;
r=1e-3; h = 1e-4;

// First base rectangle around the nozzle
Point(2) = {R, 0, 0,   20*r};
Point(3) = {R, H, 0, 2000*r};
// Second (largest) base rectangle
Point(5) = {L, 0, 0,  250*r};
Point(6) = {L, H, 0, 1000*r};
// Actual nozzle
Point(11) = {R, R,   0, r};
// Most refined area (left)
Point(14) = {	 R,  .99*R,   0, r};
Point(15) = {	 R, 1.01*R,   0, r};
// Most refined area (center)
Point(18) = {1.1*R,  .99*R,   0, r};
Point(19) = {1.1*R, 1.01*R,   0, r};
// Less refined area (left)
Point(22) = {	R,  .7*R,   0,  10*r};
Point(23) = {	R, 1.1*R,   0,  10*r};
// Less refined area (center)
Point(26) = { 30*R,  .2*R,   0, 300*r};
Point(27) = { 30*R,   4*R,   0, 500*r};

// Largest Loop
Line(2)  = {2,  5};
Line(6)  = {6,  3};
Line(12) = {23,27};
Line(13) = {27,26};
Line(14) = {26,22};
// Smaller Loop
Line(22) = {14,18};
Line(23) = {18,19};
Line(24) = {19,15};
// Verticals
Line(30) = {2, 22};
Line(31) = {22,14};
Line(32) = {14,11};
Line(33) = {11,15};
Line(34) = {15,23};
Line(35) = {23, 3};
Line(36) = {5, 6};

// Surfaces
Line Loop(4) =   {2, 36,  6,-35, 12, 13, 14,-30};
Line Loop(5) = {-14,-13,-12,-34,-24,-23,-22,-31};
Line Loop(6) =  {22, 23, 24,-33,-32};

For i In {4:6}
	Plane Surface(i) = {i};
EndFor
