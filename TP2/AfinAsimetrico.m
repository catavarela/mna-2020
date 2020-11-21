function U = AfinAsimetrico(h, U, k, q)
  % This function calculates the next U using
  % Afin integrator asymmetric with sequential computing.

  gammas = GamasAsimetrico(q);
  Z = 0;
  for i = 1:q
    X = U;
    for j = 1:i
      X = AfinMas(h/i, X, k);
    end
    Z = Z + gammas(i) .* X;
  end
  U = Z;
end