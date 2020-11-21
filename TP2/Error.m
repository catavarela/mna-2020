function E = Error(ua, ub)
  % This function calculates the error, given 2 discretized functions
  
  E = norm(Difference(ua, ub), "inf");

end