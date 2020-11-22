function U = NeriIntegrator(delta_t, U, k, q)
  a1=1/(2*(2-2^(1/3)));
  a4=a1;
  a2= - (2^(1/3) - 1)/(2*(2-2^(1/3)));
  a3=a2;
  b1=1/(2-2^(1/3));
  b3=b1;
  b4=0;
  b2=-(2^(1/3))/(2-2^(1/3));


  U = NoLineal(b4*delta_t,Lineal(a4*delta_t, NoLineal(b3*delta_t, Lineal(a3*delta_t, NoLineal(b2*delta_t, Lineal(a2*delta_t, NoLineal(b1*delta_t, Lineal(a1*delta_t, U, k), k), k), k), k), k), k), k);

end