// dimensions
l2=40; 					
l1=10; 					
h1=10;
l2b=5; 					
l1b=2; 					
h1b=1;
h1bb=2;
r=0.5;
// resolutions
r1=0.5;
r2=0.15;
r3=0.05;
r4=0.025;

// upper half
Point(1) = {-l1,h1,0,r1};
Point(2) = {l2,h1,0,r1};
Point(3) = {l2,h1bb,0,r2};
Point(4) = {-l1,h1bb,0,r2};
Point(5) = {l2,0,0,r2};
Point(6) = {l2b,0,0,r3};
Point(7) = {l2b,h1b,0,r3};
Point(8) = {-l1b,h1b,0,r3};
Point(9) = {-l1b,0,0,r3};
Point(10)= {-l1,0,0,r2};

Point(11) = {0,0,0,r3}; // center of cylinder
Point(12) = {r,0,0,r4};
Point(13) = {0,r,0,r4};
Point(14) = {-r,0,0,r4};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line(5) = {4,10};
Line(6) = {10,9};
Line(7) = {9,8};
Line(8) = {8,7};
Line(9) = {7,6};
Line(10)= {6,5};
Line(11)= {5,3};
Line(12)= {6,12};
Line(13)= {14,9};

Circle(14) = {12,11,13};
Circle(15) = {13,11,14};
Line Loop(1) = {1,2,3,4};
Line Loop(2) = {5,6,7,8,9,10,11,3};
Line Loop(3) = {7,8,9,12,14,15,13};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Physical Surface(1) = {1,2,3};
Physical Line(1) = {4,5};  //inflow
Physical Line(2) = {14,15};//cylinder
Physical Line(3) = {11,2}; //outflow
Physical Line(4) = {1};    //lateral
Physical Line(5) = {6,13,12,10}; //symmetry
