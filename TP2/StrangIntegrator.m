function U = StrangIntegrator(delta_t, U, k, q)
  U = Lineal( delta_t/2, NoLineal(delta_t, Lineal(delta_t/2,  U, k), k), k);
  end