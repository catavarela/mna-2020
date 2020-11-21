function U = AfinMas(h, U, k)
  % This function is a helper to calculate symmetric
  % Afin integrator.
  
  U = NoLineal(h, Lineal(h, U, k), k);
  
  end