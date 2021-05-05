clear;
N = 64;
n = -N:N;            % filter coefficients 
n_filter = 129;      % no. of filters
F = 0:1/n_filter:1;
BW=zeros(128,1);      % Bandwidth
CF=zeros(128,1);      % centre frequency

for i = 1:n_filter-1
    fmin = F(i);
fmax = F(i+1);
Wn = [fmin,fmax];
CF(i,1)=(fmin+fmax)/2;
BW(i,1)=fmax-fmin;

W(:,i) = hamming(length(n))'.*(sinc(Wn(2)*n).*Wn(2) - sinc(Wn(1)*n).*Wn(1));
end
Sinc_filters=W';





