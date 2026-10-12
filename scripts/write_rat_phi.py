from padephi_s import padephi_s

# compute poles for the exponential
s = 0

# choose method
method = 'pade' # %method = 'cram'

filename = "%s_poles.txt" % method

with open(filename, "w", encoding="utf-8") as file:

    for degree in range(2, 16+1, 2):

        if method == 'cram':
            raise NotImplementedError()
            # [z, c, r_inf] = cf_rat_phi(degree, s);
            # fprintf(fid, "'cram_%d':\n", degree)
        elif method == 'pade':
            # numerator and denominator degree
            n = degree
            # walk over relevant m, without duplicates
            for m in list(dict.fromkeys([0, degree-2, degree-1, degree])):

                r, a, b, z, c, r_inf = padephi_s(m, n, s=s)
                zc_sorted = sorted(zip(z, c), key=lambda zc: zc[0].imag)
                z, c = zip(*zc_sorted)
                file.write("'pade_%d_%d':\n" % (m, n))

                file.write("([ ")
                lines = []
                for k in range(len(z) // 2):
                    lines.append("% 1.20e%+1.20ej,\n   " % (z[k].real, z[k].imag))
                file.writelines(lines)

                file.write("],\n [ ")
                lines = []
                for k in range(len(c) // 2):
                    lines.append("% 1.20e%+1.20ej,\n   " % (c[k].real, c[k].imag))
                file.writelines(lines)

                file.write("],\n")
                file.write(" % 1.20e),\n" % r_inf.real)

#  '2':
#       ([ 0.1867627358357128,
#          1.3048745815331186 ],
#        [ 0.020796803046292*0.1867627358357128/2,
#         -1.443886338474624*1.3048745815331186/2],
#        0.),
