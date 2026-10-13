% official baseline steps (Challenge2_1_DMD.m) on the first 3 test sequences
% usage: matlab -batch "out_file='dmd_matlab.h5'; run('scripts/challenge/dmd_check.m')"; then
%        python scripts/challenge_eval.py --check-dmd dmd_matlab.h5
addpath('/home/HDD/cmorton/carrier_dls/papers/challenge_2.1/DMD/utils');
d = '/home/HDD/cmorton/carrier_dls/papers/challenge_2.1/data/';
ux = h5read([d 'Challenge2_1_test_input.h5'],'/ux'); uy = h5read([d 'Challenge2_1_test_input.h5'],'/uy');
uxt = h5read([d 'Challenge2_1_train.h5'],'/ux'); uyt = h5read([d 'Challenge2_1_train.h5'],'/uy');
ux_mean = mean(uxt,3); uy_mean = mean(uyt,3); clear uxt uyt
ux = ux - ux_mean; uy = uy - uy_mean;
[nx,ny,~,~] = size(ux); disp([nx ny]);
out = zeros(2*nx*ny, 30, 3);
for i = 1:3
  Q = [reshape(ux(:,:,:,i),[nx*ny 70]); reshape(uy(:,:,:,i),[nx*ny 70])];
  [Psi,~,L,~] = dmd(Q,6.25e-5,25); b = Psi\Q(:,end);
  for ti = 1:30, out(:,ti,i) = real(Psi*(b.*(L.^ti))); end
end
h5create(out_file,'/out',size(out)); h5write(out_file,'/out',out);
disp('done');
