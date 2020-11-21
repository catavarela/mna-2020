function U = AfinAsimetricoParalelo(h, U, k, q)
% This function calculates the next U using
% Afin integrator asymmetric with parallel computing.
x = U;
gammas = GamasAsimetrico(q);
n = floor(q/2); %we calculate it asymmetrically
spmd(q)
    for s = 1:n
        if labindex == s 
            for it = 1:labindex
                x = NoLineal(h/labindex, Lineal(h/labindex, x, k), k);
            end
            x = gammas(labindex) .* x;
        end

    end
    
end

for s = 2:q
    x{1} = x{1} +x{s};
end

U = x{1};

end