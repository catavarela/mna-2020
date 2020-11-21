function U = Neri(h, U, k, q)
  % This function calculates the next U using
  % Neri's order 4 symplectic integrators.
  
  a1=1/(2*(2-2^(1/3)));
  a4=a1;
  a2= - (2^(1/3) - 1)/(2*(2-2^(1/3)));
  a3=a2;
  b1=1/(2-2^(1/3));
  b3=b1;
  b4=0;
  b2=-(2^(1/3))/(2-2^(1/3));

  U = NoLineal(b4*h,Lineal(a4*h, NoLineal(b3*h, Lineal(a3*h, NoLineal(b2*h, Lineal(a2*h, NoLineal(b1*h, Lineal(a1*h, U, k), k), k), k), k), k), k), k);

end