function U = LieTrotterIntegrator(delta_t, U, k, q)
U = NoLineal(delta_t, Lineal(delta_t, U, k), k);
end