N = 256;

tic
IntStart = 0;
IntFin = 32 * pi;

x = IntervalDivider(IntStart, IntFin,N); 
d_t = 0.01;
d_x = x(2) - x(1);
d_k = (2*pi)/(N*(x(2)-x(1)));
k = [0:N/2-1 0 -N/2+1:-1]' * d_k;

q=4; frames = 10;
tensoruu = {}; tensortt = {};

for i = 1:frames
  [tt, uu] = Solver(d_t,x,k,q,1,@LieTrotter);
  tensoruu = [tensoruu uu];
  tensortt = [tensortt tt];
end

h = figure;
filename = 'Movement.gif';
for i = 1:frames
  tt = tensortt{i};
  uu = tensoruu{i};

  % ---------------- Plotting -----------------------------
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
% end
toc
delete(gcp('nocreate'));
