function U = AffineSymmetricParallel(delta_t, U, k, q)

  X = U;

  m = floor(q/2);

  gamas = SymmetricGamas(q);

  spmd(q)

     for j = 1:m

      if r == j 

        for i = 1:r
          X = NoLineal(delta_t/r, Lineal(delta_t/r, X, k), k); %I-
        end

        X = gamas(r) .* X;

      end
      
      if r == j + m  

         for i = 1:r-m 
             X= NoLineal(delta_t/(r-m), Lineal(delta_t/(r-m), X, k), k); %I+
         end

         X = gamas(r-m) .* X; 

      end
       
     end
  end

  for i = 2:m
      X{1} = X{1} +X{i};
  end

  U = X{1};
  
end