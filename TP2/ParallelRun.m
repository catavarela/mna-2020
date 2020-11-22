% Analysis constants
N = 256; %number of discrete

% interval defined
IntStart = 0;
IntFin = 32 * pi;
x = IntervalDivider(IntStart, IntFin,N); %discretization of interval
d_t = 0.002;
d_x = x(2) - x(1);
d_k = (2*pi)/(N*(x(2)-x(1)));
k = [0:N/2-1 0 -N/2+1:-1]' * d_k;
q = 2;
c = parcluster;
c.NumWorkers = q;
parpool('local', q);

[tt, uu] = ParalelSolver(d_t, x,k,q,0,@AfinSimetricoParalelo);

surf(tt,x,uu), shading interp, lighting phong, axis tight
view([-90 90]), colormap(autumn); set(gca,'zlim',[-5 50])
light('color',[1 1 0],'position',[-1,2,2])
material([0.30 0.60 0.60 40.00 1.00]);
