function X = CropInterval(s, f, N)

X = (s+(f/N):(f - s)/N:f)';

end