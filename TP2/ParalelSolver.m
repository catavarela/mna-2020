  % solves the differential equation using a parallel integrator.

  % h: time step
  % x: values of x
  % k: values of k
  % q: order of the integrator
  % p: perturbance enabled or disabled
  

function [tt, uu] = ParalelSolver(h,x,k,q,p)

  tmax = 150;

  nmax = round(tmax / h);

  nplt = floor((tmax / 100) / h);

  perturbance = x * (rand * 0.01 - 0.005) * p; % p is 1 if perturbance is enabled, 0 otherwise

  px = x + perturbance;



  u = InitialCondition(px);
  
  U = fft(u);

  uu = u;
  tt = 0;
  for n = 1:nmax
      t = n * h;

      U = AfinAsimetricoParalelo(h, U, k, q);

      if mod(n, nplt) == 0
        u = real(ifft(U));
        uu = [uu, u]; tt = [tt, t];
      end
  end
end