function U = NoLineal(delta_t, U, k)
% This function calculates the Non-Linear part
% given a function, a delta_t and the values k.
% using RK4

g = -(1/2)*1i*delta_t*k;
a = g.*fft(real(ifft(U)).^2);
b = g.*fft(real(ifft(U + a/2)).^2);
c = g.*fft(real(ifft(U + b/2)).^2);
d = g.*fft(real(ifft(U + c)).^2);
U = U + (a + 2*(b+c) + d)/6;

end