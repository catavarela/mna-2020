function E = Error(x, y)

  E = norm(Difference(x, y), 'inf');

end