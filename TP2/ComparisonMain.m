% Analysis constants
N = 256;
x = IntervalDivider(0, 32*pi, N);
h = 0.002;
k = [0:N/2-1 0 -N/2+1:-1]' / 16;

% Render constants
frames = 5;
tensoruu = {};
tensortt = {};
errors = [];
% Solver method
% for i = 1:frames
for i = 1:100
  [tt, uu] = ComparisonSolver(h * i,x,k,4,0, @AfinAsimetrico);
  [tt2, uu2] = ComparisonSolver((h * i)/2,x,k,4,0, @AfinAsimetrico);
  tt2 = tt2(1:2:end);
  uu2 = uu2(:, 1:2:end);

  [m,n] = size(uu);
  [m2,n2] = size(uu2);
  if n < n2
    uu2=uu2(:,1:end-1);
    tt2=tt2(:,1:end-1);
  end
  if n > n2
    uu=uu(:,1:end-1);
    tt=tt(:,1:end-1);
  end


  Error(tt, tt2)
  errors = [errors Error(uu, uu2)];
%   tensoruu = [tensoruu uu];
%   tensortt = [tensortt tt];
end

output = [[1:100] * h;errors];
dlmwrite('myFile.txt',output,'delimiter','\t')
% end

% Render cycle
% while true
% for i = 1:frames
%   tt = tensortt{i};
%   uu = tensoruu{i};

%   % Plot results:
%   surf(tt, x, uu), shading interp, lighting phong, axis tight
%   view([-90 90]), colormap(autumn); set(gca, 'zlim', [-5 50])
%   light('color', [1 1 0], 'position', [-1, 2, 2])
%   material([0.30 0.60 0.60 40.00 1.00]);
%   pause(0.033);
% end
% end