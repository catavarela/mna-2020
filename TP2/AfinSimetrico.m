function U = AfinSimetrico(h, U, k, q)
  % This function calculates the next U using
  % Afin integrator symmetric with sequential computing.

  gammas = GamasSimetrico(q);
  Z = 0;
  n = q/2;
  for i = 1:n
    X = U;
    Y = U;
    for j = 1:i
      X = AfinMas(h/i, X, k);
      Y = AfinMenos(h/i, Y, k);
    end
    Z = Z + gammas(i) .* X + gammas(i) .* Y;
  end
  U = Z;
end