load('AM_S=1.000_m=-1.mat')

vals=[];
for re=linspace(-.05,.05,5)
    for im=linspace(-1,1,5)
        try
            [V,D]=eigs(-A,M,10,re+1j*im);
            vals=[vals diag(D)];
        catch WarnIfIllConditioned
            pass
        end
    end
end

plot(imag(vals),real(vals),'*')

save('validation_S=1.000_m=-1.mat')
