function y = customFFT(x)
n = length(x);

if mod(n,2) == 0
    dom = (0:n/2-1);
    e = exp(-2i*pi/n) .^ dom;
    odd = customFFT(x(1:2:n-1));
    even = e.*customFFT(x(2:2:n));
    y = [odd+even, odd-even];
else
    dom_t = 0:n-1;
    dom = dom_t';
    F = exp(-2i*pi/n) .^ (dom*dom_t);
    y = F*x;
end