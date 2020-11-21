function U = AfinMenos(h, U, k)
  % This function is a helper to calculate symmetric
  % Afin integrator.
  
  U = Lineal(h, NoLineal(h, U, k), k);
  
  end