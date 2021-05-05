n=-64:64;% filter coefficients
p = 128; % no. of filterbanks
fs = 16000;
NFFT = 512;
f=fs/2*linspace(0,1,NFFT/2+1);
fmel=2595*log10(1+f./700); % CONVERTING TO MEL SCALE
fmelmax=max(fmel);
fmelmin=min(fmel);
filbandwidthsmel=linspace(fmelmin,fmelmax,p+2);
filbandwidthsf=700*(10.^(filbandwidthsmel/2595)-1);
melfreq = filbandwidthsf(2:end-1);
F=filbandwidthsf;
F=F/8000;
CF_imel=zeros(p,1);
BW_imel=zeros(p,1);
% inverse mel scale
F = abs(fliplr(F)-1); %to invert inverse mel scale

for i = 1:p
    fmin = F(i);
fmax = F(i+2);
Wn = [fmin,fmax];
CF_imel(i,1)=(fmin+fmax)/2;
BW_imel(i,1)=fmax-fmin;
W(:,i) = hamming(length(n))'.*(sinc(Wn(2)*n).*Wn(2) - sinc(Wn(1)*n).*Wn(1));
end
Sinc_filters=W';
