function [zi, ci, r_inf] = cf_rat_phi(n, l)
  K = 75; % no of Cheb coeffs
  nf = 1024; % no of pts for FFT
  w = exp(2i*pi*(0:nf-1)/nf); % roots of unity
  t = real(w); % Cheb pts (twice over)
  scl = 9; % scale factor for stability
  F = zeros(size(t));
  g = (t~=-1);
  F(g) = phiS(scl*(t(g)-1)./(t(g)+1),l); % evaluate Phi_l on neg. axis
  c = real(fft(real(F)))/nf; % Cheb coeffs of F
  f = polyval(c(K+1:-1:1),w); % analytic part f of F
  [U,S,V] = svd(hankel(c(2:K+1))); % SVD of Hankel matrix
  s = S(n+1,n+1); % singular value
  u = U(K:-1:1,n+1)'; v = V(:,n+1)'; % singular vectors
  zz = zeros(1,nf-K); % zeros for padding
  b = fft([u zz])./fft([v zz]); % finite Blaschke product
  rt = f-s*w.^K.*b; % extended function r-tilde
  rtc = real(fft(rt))/nf; % its Laurent coeffs
  zr = roots(v); qj = zr(abs(zr)>1); % poles
  qc = poly(qj); % coeffs of denominator
  pt = rt.*polyval(qc,w); % numerator
  ptc = real(fft(pt)/nf); % coeffs of numerator
  ptc = ptc(n+1:-1:1); ci = 0*qj;

  for k = 1:n % calculate residues
    q = qj(k);
    q2 = poly(qj(qj~=q));
    ci(k) = polyval(ptc,q)/polyval(q2,q);
  end
  zi = scl*(qj-1).^2./(qj+1).^2; % poles in z-plane
  ci = 4*ci.*zi./(qj.^2-1); % residues in z-plane
  r_inf = 0.5*(phiS(0,l)+sum(ci./zi)); % r at infinity
  [m, order]=sort(imag(zi)); % sort poles
  zi=zi(order); % reorder poles
  ci=ci(order); % reorder residues
end

function phi_s_Z = phiS(Z, S)

%compute phi_s(Z), reliably up to S=10 or so

[nx, mx] = size(Z);
z = Z(:);

tayl = @(z, N, s) sum(z.^(0:N) ./ factorial((0:N)+s), 2);

Is = (abs(z) < 1);
if any(Is)
    phi_s_Z(Is) = tayl(z(Is), 20+S, S);
end
if any(~Is)
    phi_s_Z(~Is) = (exp(z(~Is)) - tayl(z(~Is), S-1, 0)) ./ z(~Is).^S;
end

phi_s_Z = reshape(phi_s_Z, nx, mx);

end
  
