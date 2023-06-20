[X,Y,Z]=meshgrid([-0.7:0.05:0.7],[-0.7:0.05:0.7],[0:pi/30:2*pi]);
Theta = atan2(Y,X);
R = sqrt(X.^2+Y.^2);

k = 2;
m = -2;
omega = 1;

nt = 40;
M(nt) = struct('cdata',[],'colormap',[]);

% tiltview=false shows the true result of the computations.
% tiltview=true switches the axes for a better view, with potential for
% human error
tiltview = false;

for it = 1:nt
    clf
    t = abs(it*2*pi/nt/omega);
    V = real(exp(-(R-0.5).^2/0.05).*exp(1i*(k*Z + m*Theta - omega*t)));

    if tiltview
        s=isosurface(Z,X,Y,V,0.6);
    else
        s=isosurface(X,Y,Z,V,0.6);
    end
    p=patch(s);
    set(p,'FaceColor',[0.25 0.25 1],'EdgeColor','none');  
   
    if tiltview
        s=isosurface(Z,X,Y,V,-0.6);
    else
        s=isosurface(X,Y,Z,V,-0.6);
    end
    p=patch(s);
    set(p,'FaceColor',[1 0.25 0.25],'EdgeColor','none');
    
    view(3);
    if tiltview
        axis([0 6.5 -0.7 0.7 -0.7 0.7])
    else
        axis([-0.7 0.7 -0.7 0.7 0 6.5])
    end
    camlight;
    lighting gouraud;
    daspect([1,1,1])

    drawnow
    M(it) = getframe;
end