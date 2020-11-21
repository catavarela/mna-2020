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
      Y = Lineal(h/i, NoLineal(h/i, Y, k), k); %I-
      X =  NoLineal(h/i, Lineal(h/i, X, k), k); %I+
    end
    Z = Z + gammas(i) .* X + gammas(i) .* Y;
  end
  U = Z;
end
