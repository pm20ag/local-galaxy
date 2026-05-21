k = 1.38*10^(-23);
muzero = 1.2566*10^-6;
Av = 6.02*10^(23);
%B = muzero*I/2*pi*r
%Pm = B^2/2*muzero
%Pp = n*k*T

m = 1;
Mm = 2.0141;
moles = m/Mm;
atoms = moles*Av;
ions = atoms;
electrons = atoms;
volume = pi*(0.5)^2*(10);
n_i = ions/volume;
n_e = electrons/volume;
T = 1000;
Pp = (n_i + n_e)*k*T;

prior_mean = 125000^2;
prior_sd = 3000^2;
single_likelihood_sd = 30;
r_0 = 0.5;
M = 0.25;
omega = 2;
Observations = 1000;

Pm = zeros(1, Observations);

i = 1;
S_1 = 0;
S_2 = 0;
while i <= Observations
    Pm(i) = Pp + single_likelihood_sd*randn(); %1050
    S_1 = S_1 + 1/(((8*pi^2*(r_0 + M*sin(omega*i))^2)/muzero)^2*single_likelihood_sd^2);
    S_2 = S_2 + Pm(i)/(((8*pi^2*(r_0 + M*sin(omega*i))^2)/muzero)*single_likelihood_sd^2);
    i = i + 1;
end

posterior_varience = 1/(1/((prior_sd)^2) + S_1);
posterior_mean = posterior_varience*(prior_mean/((prior_sd)^2) + S_2);
posterior_sd = sqrt(posterior_varience);

Montes = 1000;
I_square = zeros(1, Montes);
j = 1;
S_3 = 0;
while j <= Montes
    I_square(j) = posterior_mean + posterior_sd*randn();
    S_3 = S_3 + sqrt(I_square(j));
    j = j + 1;
end

mean_current = S_3/Montes;
