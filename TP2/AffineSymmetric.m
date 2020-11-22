function U = AffineSymmetric(delta_t, U, k, q)
  
  Z = 0;

  m = floor(q/2);

  gamas = SymmetricGamas(q);
  
  for j = 1:m
    X = U;
    Y = U;
    
    for it = 1:j
      Y = Lineal(delta_t/j, NoLineal(delta_t/j, Y, k), k); 
      X =  NoLineal(delta_t/j, Lineal(delta_t/j, X, k), k); 
    end
    
    Z = Z + gamas(j) .* X + gamas(j) .* Y;
  
  end
  
  U = Z;

end
