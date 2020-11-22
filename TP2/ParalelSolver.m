  % solves the differential equation using a parallel integrator.

  % h: time step
  % x: values of x
  % k: values of k
  % q: order of the integrator
  % p: perturbance enabled or disabled
  

function [tt, uu] = ParalelSolver(d_t,x,k,q,p,integrator)

  tmax = 150;

  nmax = round(tmax / d_t);

  nplt = floor((tmax / 100) / d_t);

  perturbance = x * (rand * 0.01 - 0.005) * p; % p is 1 if perturbance is enabled, 0 otherwise

  px = x + perturbance;



  u = InitialCondition(px);
  
  U = fft(u);

  uu = u;
  tt = 0;
  for n = 1:nmax
      t = n * d_t;

      U = integrator(d_t, U, k, q);

      if mod(n, nplt) == 0
        u = real(ifft(U));
        uu = [uu, u]; tt = [tt, t];
      end
  end
end