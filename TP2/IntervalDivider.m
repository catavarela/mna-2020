function x = IntervalDivider(IntStart, IntFin, N)
%this function divides the interval into N pieces

res = (IntStart+(IntFin/N):(IntFin - IntStart)/N:IntFin)';

x = res;

end