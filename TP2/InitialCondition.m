function U = InitialCondition(x)
  % This function calculates returns the initial condition
  % for the differential equation
  
  U = cos(x / 16) .* (1 + sin(x / 16));
  
  end