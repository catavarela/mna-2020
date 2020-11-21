function gammas = GamasAsimetrico(q)
  % This function calculates the gammas for the asymmetric Afin integrator

  M = ones(q);
  for j = 1:q
    for i = 1:q-1
      M(j,i) = i + M(j,i);
    end
  end
  
  for j = 1:q
    for i = 1:q
      M(j,i) = M(j,i).^(1-j);
    end
  end
  
  gammas = inv(M) * [1 zeros(1, q - 1)]';
end