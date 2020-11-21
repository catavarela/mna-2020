function U = NoLinealEuler(delta_t, U, k)
  % This function calculates the Non-Linear part
  % given a function, a delta_t and the values k.
  % using Euler
  
  U = U  - ((1/2)*1i*k*delta_t).*fft((real(ifft(U))).^2);

  end