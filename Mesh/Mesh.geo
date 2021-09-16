Mesh.MshFileVersion = 2.0;
// dimensions
r=0.5;		// cylinder
l1up=-1.;  	// first (inner) rectangle			
l1down=3.;					
h1=1.;
l2up=-3.;	// second rectangle
h2=2.;
l3up = -10.;	// outer rectangle (box boundary)
l3down = 30.;	
h3 = 10.;

// resolutions
rc=0.025;
r1=0.1;
r2=0.2;
r3=1.;

// cylinder
Point(1) = {0,0,0,rc}; // center
Point(2) = {r,0,0,rc};
Point(3) = {0,r,0,rc};
Point(4) = {-r,0,0,rc};
Point(5) = {0,-r,0,rc};
// first (inner) rectangle
Point(11) = {l1up,h1,0,r1};
Point(12) = {l1down,h1,0,r1};
Point(13) = {l1down,-h1,0,r1};
Point(14) = {l1up,-h1,0,r1};
// second rectangle
Point(21) = {l2up,h2,0,r2};
Point(22) = {l3down,h2,0,r2};
Point(23) = {l3down,-h2,0,r2};
Point(24) = {l2up,-h2,0,r2};
// outer rectangle (box boundary)
Point(31) = {l3up,h3,0,r3};
Point(32) = {l3down,h3,0,r3};
Point(33) = {l3down,-h3,0,r3};
Point(34) = {l3up,-h3,0,r3};

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};
// first (inner) rectangle
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,11};
// second rectangle
Line(21) = {21,22};
Line(22) = {22,23};
Line(23) = {23,24};
Line(24) = {24,21};
// outer rectangle (box boundary)
Line(31) = {31,32};
Line(32) = {32,22};
Line(33) = {23,33};
Line(34) = {33,34};
Line(35) = {34,31};

Line Loop(1) = {1,2,3,4};
Line Loop(2) = {11,12,13,14};
Line Loop(3) = {21,22,23,24};
Line Loop(4) = {31,32,-21,-24,-23,33,34,35};
Plane Surface(1) = {1,2};	// inner rectangle minus the cylinder
Plane Surface(2) = {2,3};	// second rectangle minus inner rectangle
Plane Surface(3) = {4};		// outer rectangle minus second rectangle
Physical Surface(1) = {1,2,3};  // add all to form one physical domain
Physical Line(1) = {1,2,3,4};	// cylinder
Physical Line(2) = {35};	// inlet
Physical Line(3) = {31,34};	// lateral boundaries
Physical Line(4) = {32,22,33};	// outlet


