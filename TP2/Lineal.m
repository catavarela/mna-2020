function U = Lineal(delta_t, U, k)
U = U.*exp((k.^2 - k.^4)*delta_t);
end