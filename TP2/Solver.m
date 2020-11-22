  % h: time step
  % x: values of x
  % k: values of k
  % q: order of the integrator
  % p: perturbance enabled or disabled

function [tt, uu] = Solver(delta_t,x,k,q,p,integrator)
  
  tmax = 150;
  nmax = round(tmax / delta_t);
  nplt = floor((tmax / 100) / delta_t);
  perturbance = x * (rand * 0.01 - 0.005) * p; 
  px = x + perturbance;

  u = InitialCondition(px);

  U = fft(u);


  uu = u;
  tt = 0;
  for n = 1:nmax
      t = n * delta_t;

      U = integrator(delta_t, U, k, q);

      if mod(n, nplt) == 0
        u = real(ifft(U));
        uu = [uu, u]; tt = [tt, t];
      end
  end
end