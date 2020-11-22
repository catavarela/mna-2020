function gamas = AsymmetricGamas(q)

  matrix = ones(q);

  for i = 1:q

    for j = 1:q-1
      matrix(i,j) = matrix(i,j) + j;
    end

  end
  
  for i = 1:q

    for j = 1:q
      matrix(i,j) = matrix(i,j).^(1-i);
    end

  end
  
  gamas = inv(matrix) * [1 zeros(1, q - 1)]';
  
end