function U = AffineAsymmetric(delta_t, U, k, q)
  
  Z = 0;
  
  gamas = AsymmetricGamas(q);
  
  for j = 1:q
    X = U;

    for i = 1:j
      X = NoLineal(delta_t/j, Lineal(delta_t/j, X, k), k);
    end
    
    Z = Z + gamas(j) .* X;
  end
  
  U = Z;

end