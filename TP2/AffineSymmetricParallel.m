function U = AffineSymmetricParallel(delta_t, U, k, q)

  X = U;

  m = floor(q/2);

  gamas = SymmetricGamas(q);

  spmd(q)

     for j = 1:m

      if labindex == j 

        for i = 1:labindex
          X = NoLineal(delta_t/labindex, Lineal(delta_t/labindex, X, k), k); %I-
        end

        X = gamas(labindex) .* X;

      end
      
      if labindex == j + m  

         for i = 1:labindex-m 
             X= NoLineal(delta_t/(labindex-m), Lineal(delta_t/(labindex-m), X, k), k); %I+
         end

         X = gamas(labindex-m) .* X; 

      end
       
     end
  end

  for i = 2:q
      X{1} = X{1} +X{i};
  end

  U = X{1};
  
end