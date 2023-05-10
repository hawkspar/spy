// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.2;
R=1; X=1.2;
L=50*R; H=10*R;
r=1e-3; h = 1e-4;

// First base rectangle around the nozzle
Point(2) = {X, 0, 0,   20*r};
Point(3) = {X, H, 0, 1500*r};
// Second (largest) base rectangle
Point(5) = {L, 0, 0,  250*r};
Point(6) = {L, H, 0, 1000*r};
// Actual nozzle
Point(11) = {X, R,   0, r};
// Less refined area (left)
Point(22) = {X,  .7*R,   0,  10*r};
Point(23) = {X, 1.1*R,   0,  10*r};
// Less refined area (center)
Point(26) = {30*R, .25*R,   0, 200*r};
Point(27) = {30*R,   6*R,   0, 500*r};

// Largest Loop
Line(2)  = {2,  5};
Line(6)  = {6,  3};
Line(12) = {23,27};
Line(13) = {27,26};
Line(14) = {26,22};
// Verticals
Line(30) = {2, 22};
Line(32) = {22,11};
Line(33) = {11,23};
Line(35) = {23, 3};
Line(36) = {5, 6};

// Surfaces
Line Loop(1) =   {2, 36,  6,-35, 12, 13, 14,-30};
Line Loop(2) = {-14,-13,-12,-33,-32};

For i In {1:2}
	Plane Surface(i) = {i};
EndFor
