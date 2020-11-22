function gamas = SymmetricGamas(q)

  m = q/2;
  matrix = ones(m);

  for i = 1:m

    for j = 1:m-1
      matrix(i,j) = matrix(i,j) + j;
    end

  end
  
  for i = 1:m

    for j = 1:m
      matrix(i,j) = matrix(i,j).^(1-i);
    end

  end
  
  gamas = inv(matrix) * [1/2 zeros(1, m - 1)]';

end