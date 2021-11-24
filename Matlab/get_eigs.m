load('AM_S=1.000_m=-1.mat')

[V,D]=eigs(-A,M,15,.0+1.2j);

D=diag(D);
plot(imag(D),real(D),'*')

save('validation_S=1.000_m=-1.mat')
