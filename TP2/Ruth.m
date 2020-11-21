function U = Ruth(h, U, k, q)
  % This function calculates the next U using
  % Ruth's order 3 symplectic integrators.
  
  a1=7/24;
  a2=3/4;
  a3=-1/24;
  b1=2/3;
  b2=-2/3;
  b3=1;

  U = NoLineal(b3*h, Lineal(a3*h, NoLineal(b2*h, Lineal(a2*h, NoLineal(b1*h, Lineal(a1*h, U, k), k), k), k), k), k);

end