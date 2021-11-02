// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.0;
r1 = 1/20;
r4 = 2;
r3=(r1+r4)/2;
r2=(r1+r3)/2;
Point(1) = {0, 0, 0, r1};
Point(2) = {50, 0, 0, r1};
Point(3) = {50, 1, 0, r2};
Point(4) = {0, 1, 0, r1};
Point(5) = {0, 5, 0, r3};
Point(6) = {70, 5, 0, r3};
Point(7) = {70, 10, 0, r3};
Point(8) = {0, 10, 0, r3};
Point(9) = {70, 0, 0, r2};
Point(10) = {120, 0, 0, r3};
Point(11) = {120, 60, 0, r4};
Point(12) = {0, 60, 0, r4};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {4, 5};
Line(10) = {6, 9};
Line(11) = {9, 10};
Line(12) = {10, 11};
Line(13) = {11, 12};
Line(14) = {12, 8};
Line(15) = {2, 9};
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {-15, 10, 5, 9, 3, 2};
Line Loop(3) = {5, 6, 7, 8};
Line Loop(4) = {11, 12, 13, 14, -7, -6, 10};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Physical Line(1) = {1,15,11};
Physical Line(2) = {12};
Physical Line(3) = {13};
Physical Line(4) = {14,8,9,4};
Physical Surface(1) = {1,2,3,4};
