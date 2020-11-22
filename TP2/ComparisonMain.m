N = 256;
x = CropInterval(0, 32*pi, N);
h = 0.002;
k = [0:N/2-1 0 -N/2+1:-1]' / 16;

frames = 5;
tensoruu = {};
tensortt = {};
errors = [];

for i = 1:100
  [tt, uu] = ComparisonSolver(h * i,x,k,4,0, @AffineAsymmetric);
  [tt2, uu2] = ComparisonSolver((h * i)/2,x,k,4,0, @AffineAsymmetric);
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

end

output = [[1:100] * h;errors];
dlmwrite('myFile.txt',output,'delimiter','\t')

