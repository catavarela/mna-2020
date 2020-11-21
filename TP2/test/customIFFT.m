frec = -50:1/10:50;
Xf = sinc(frec);

figure;
subplot(2,1,1);
plot(frec, Xf);

time = -10:1/50:10;
xt_1 = ifft(Xf);
xt_2 = customIFFT(Xf);

subplot(2,1,2);
plot(time, abs(fftshift(xt_1))); hold on;
plot(time, abs(fftshift(xt_2)));
