function U = AffineAsymmetricParallel(delta_t, U, k, q)

X = U;

m = floor(q/2);

gamas = AsymmetricGamas(q);

spmd(q)

    for i = 1:m
        
        if labindex == i 
            
            for it = 1:labindex
                X = NoLineal(delta_t/labindex, Lineal(delta_t/labindex, X, k), k);
            end
            
            
        end
        X = gamas(labindex) .* X;

    end
    
end

for i = 2:q
    X{1} = X{1} +X{i};
end

U = X{1};

end