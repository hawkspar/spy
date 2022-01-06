// Gmsh project created on Fri May 15 17:21:02 2020
Mesh.MshFileVersion = 2.0;
r1 = 1/200;
r4 = 2/5;
r3=(r1+r4)/2;
r2=(r1+r3)/2;
Point(1) = {0, 0, 0, r1};
Point(2) = {10, 0, 0, r1};
Point(3) = {10, 5, 0, r2};
Point(4) = {0, 5, 0, r2};
Point(5) = {20, 0, 0, r3};
Point(6) = {20, 10, 0, r4};
Point(7) = {0, 10, 0, r3};
Point(8) = {0, 1, 0, r1};
Point(9) = {1, 1, 0, r1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 8};
Line(5) = {8, 1};
Line(6) = {2, 5};
Line(7) = {5, 6};
Line(8) = {6, 7};
Line(9) = {7, 4};
Line(10) = {8, 9};
Line Loop(1) = {1, 2, 3, 4, 10, -10, 5};
Line Loop(2) = {6, 7, 8, 9, -3, -2};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Physical Line(1) = {1,6};
Physical Line(2) = {7};
Physical Line(3) = {8};
Physical Line(4) = {9,4,5};
Physical Surface(1) = {1,2};
