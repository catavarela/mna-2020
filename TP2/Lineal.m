function U = Lineal(delta_t, U, k)
% This function calculates the Linear part analitically
% given a function, a delta_t and the values k.

U = U.*exp((k.^2 - k.^4)*delta_t);

end