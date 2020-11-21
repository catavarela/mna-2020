function time = customIFFT(frec) 
    
    N = length(frec);
    time = zeros(size(frec));
    
    for n = 1:length(time)
        time(n) = sum(frec.*exp(1i*2*pi*(0:N-1)*n/N))/N; 
    end

end
