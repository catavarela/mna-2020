% Analysis constants
N = 256; %number of discrete


tic % we measure time from this point


% method --> 1 - Lie Trotter
% method --> 2 - Strang
% method --> 3 - Ruth
% method --> 4 - Neri
% method --> 5 - Affine Symmetric
% method --> 6 - Affine Asymmetric

method = 6;
% interval defined
IntStart = 0;
IntFin = 32 * pi;
x = IntervalDivider(IntStart, IntFin,N); %discretization of interval
d_t = 0.01;
d_x = x(2) - x(1);
d_k = (2*pi)/(N*(x(2)-x(1)));
k = [0:N/2-1 0 -N/2+1:-1]' * d_k;

% Render constants
frames = 1;
tensoruu = {};
tensortt = {};

q=10; %this is only used in afin methods

% Solver method
for i = 1:frames
    if method == 1
        [tt, uu] = Solver(d_t,x,k,q,1,@LieTrotterIntegrator);
    elseif method == 2
        [tt, uu] = Solver(d_t,x,k,q,1,@StrangIntegrator);
    elseif method == 3
        [tt, uu] = Solver(d_t,x,k,q,1,@RuthIntegrator);
    elseif method == 4
        [tt, uu] = Solver(d_t,x,k,q,1,@NeriIntegrator);
    elseif method == 5
        [tt, uu] = Solver(d_t,x,k,q,1,@AffineSymmetric);
    elseif method == 6
        [tt, uu] = Solver(d_t,x,k,q,1,@AffineAsymmetric);
    end
    
  tensoruu = [tensoruu uu];
  tensortt = [tensortt tt];
end

% Gif creation
h = figure;
filename = 'Movement.gif';
for i = 1:frames
  tt = tensortt{i};
  uu = tensoruu{i};

  % Plot results:
  surf(tt, x, uu), shading interp, lighting phong, axis tight
  view([-90 90]), colormap(autumn); set(gca, 'zlim', [-5 50])
  light('color', [1 1 0], 'position', [-1, 2, 2])
  material([0.30 0.60 0.60 40.00 1.00])
  drawnow
  frame = getframe(h);
  im = frame2im(frame);
  [imind,cm] = rgb2ind(im,256);
  if i == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
  end
end


toc %we finish measuring time

delete(gcp('nocreate')); %to shutdown parpool
