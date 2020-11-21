
% Calculate the next U using the Strang Integrator

function U = Strang(h, U, k, q)
  U = Lineal( h/2, NoLineal(h, Lineal(h/2,  U, k), k), k);
  end