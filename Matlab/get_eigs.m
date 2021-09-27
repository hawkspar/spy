load('incompressible_jet.mat')

[V,D]=eigs(A*1j,M,50,0.25-0.3j);
vals=diag(D)

plot(real(vals),imag(vals),'*')

save('incompressible_jet.mat')
