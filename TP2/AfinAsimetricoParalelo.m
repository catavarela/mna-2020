function U = AfinAsimetricoParalelo(h, U, k, q)
% This function calculates the next U using
% Afin integrator asymmetric with parallel computing.

x = U;
gammas = GamasAsimetrico(q);

spmd(q)
    for j = 1:labindex
        x = AfinMas(h/labindex, x, k);
    end 
    x = gammas(labindex) * x;
end

for s = 2:q
    x{1} = x{1} +x{s};
end

U = x{1};

end