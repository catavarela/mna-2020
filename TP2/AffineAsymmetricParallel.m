function U = AffineAsymmetricParallel(delta_t, U, k, q)

X = U;

m = floor(q/2);

gamas = AsymmetricGamas(q);

spmd(q)

    for i = 1:m
        
        if j == i 
            
            for it = 1:j
                X = NoLineal(delta_t/j, Lineal(delta_t/j, X, k), k);
            end
            
            X = gamas(j) .* X;
        end

    end
    
end

for i = 2:q
    X{1} = X{1} +X{i};
end

U = X{1};

end