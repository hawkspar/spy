load('AM_S=1.000_m=-1.mat')

[V,D]=eigs(A,M,15,.05+1j);

%plot(-imag(vals),-real(vals),'*')

save('validation_S=1.000_m=-1.mat')
