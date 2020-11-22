function U = RuthIntegrator(delta_t, U, k, q)
  a1=7/24; a2=3/4; a3=-1/24; b1=2/3; b2=-2/3; b3=1;
  U = NoLineal(b3*delta_t, Lineal(a3*delta_t, NoLineal(b2*delta_t, Lineal(a2*delta_t, NoLineal(b1*delta_t, Lineal(a1*delta_t, U, k), k), k), k), k), k);
end