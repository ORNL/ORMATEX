
%% compute poles for the exponential
s = 0;

%% choose method
method = 'cram'

filename = sprintf("%s_poles.txt", method);
fid = fopen(filename, "w");

for degree = 2:2:16

  if method == 'cram'
    [z, c, r_inf] = cf_rat_phi(degree, s);
    fprintf(fid, "'cram_%d':\n", degree)
  end

  fprintf(fid, "([ ")
  for k = 1:ceil(length(z)/2)
    fprintf(fid, "% 1.14e%+1.14ej,\n   ", real(z(k)), imag(z(k)))
  end
  fprintf(fid, "],\n [ ")
  for k = 1:ceil(length(c)/2)
    fprintf(fid, "% 1.14e%+1.14ej,\n   ", real(c(k)), imag(c(k)))
  end
  fprintf(fid, "],\n")
  fprintf(fid, " % 1.14e),\n", real(r_inf))
  #  '2':
  #       ([ 0.1867627358357128,
  #          1.3048745815331186 ],
  #        [ 0.020796803046292*0.1867627358357128/2,
  #         -1.443886338474624*1.3048745815331186/2],
  #        0.),
end

fclose(fid);
