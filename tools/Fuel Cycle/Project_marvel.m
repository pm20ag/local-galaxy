Ntin = 0.00009*3600; %0.00009
t0 = 0;
tN = 2400;
t = t0; %28
h = 0.001;
steps = (tN - t0)/h + 1; %24001
samples = 1;

time = zeros(1, steps);
IOFC = zeros(steps, samples);
IIFC = zeros(steps, samples);
Istorage = zeros(steps, samples);
NIOFC = zeros(steps, samples);
NIIFC = zeros(steps, samples);
NIstorage = zeros(steps, samples);

Ntburn = zeros(steps, samples);
TBR = zeros(steps, samples);
Istartup = zeros(steps, samples);
tIFC = zeros(steps, samples);
tOFC = zeros(steps, samples);

a= 0; %0 or 1
b = 1; %0 or 1
c = 0; %0 or 1
d = 0; %0 or 1
e = 0; %0 or 1

i = 1;
j = 1;
while i <= samples
    while j <= steps
    if j == 1
        Ntburn(j, i) = 0.01*Ntin + a*0.001*Ntin*randn(); %0.0000009 + 0.00000009*randn()
        TBR(j, i) = 1.08 + b*0.03*randn();
        Istartup(j, i) = 1.5 + c*0.15*randn();
        tIFC(j, i) = 4 + d*0.4*randn();
        tOFC(j, i) = 24 + e*2.4*randn();
        NIOFC(j, i) = 0;
        NIIFC(j, i) = 0;
        NIstorage(j, i) = Istartup(j, i);
    else
        Ntburn(j, i) = Ntburn(j - 1, i);
        TBR(j, i) = TBR(j - 1, i);
        Istartup(j, i) = Istartup(j - 1, i);
        tIFC(j, i) = tIFC(j - 1, i);
        tOFC(j, i) = tOFC(j - 1, i);
    end
    
    %while j <= steps
    TBE = Ntburn(j, i)/Ntin;
    %Exact
    IOFC(j, i) = Ntburn(j, i)*tOFC(j, i)*TBR(j, i)*(1 - exp(-t/tOFC(j, i)));
    IIFC(j, i) = Ntburn(j, i)*tIFC(j, i)*((1 - TBE)/TBE)*(1 - exp(-t/tIFC(j, i))) + Ntburn(j, i)*TBR(j, i)*tIFC(j, i)*(1 - exp(-t/tIFC(j, i))) + Ntburn(j, i)*TBR(j, i)*((tIFC(j, i)*tOFC(j, i))/(tOFC(j, i) - tIFC(j, i)))*(exp(-t/tIFC(j, i)) - exp(-t/tOFC(j, i)));
    Istorage(j, i) = Istartup(j, i) + Ntburn(j, i)*(TBR(j, i) - 1)*t + Ntburn(j, i)*TBR(j, i)*((tIFC(j, i)*tIFC(j, i))/(tOFC(j, i) - tIFC(j, i)))*(1 - exp(-t/tIFC(j, i))) + Ntburn(j, i)*TBR(j, i)*((tOFC(j, i)*tOFC(j, i))/(tOFC(j, i) - tIFC(j, i)))*(1 - exp(-t/tOFC(j, i))) - Ntburn(j, i)*tIFC(j, i)*((1 - TBE)/TBE)*(1 - exp(-t/tIFC(j, i)));
    %Euler
    if j < steps
    NIOFC(j + 1, i) = NIOFC(j, i) + h*(Ntburn(j, i)*TBR(j, i) - NIOFC(j, i)/tOFC(j, i));
    NIIFC(j + 1, i) = NIIFC(j, i) + h*(((1 - TBE)/TBE)*Ntburn(j, i) + NIOFC(j, i)/tOFC(j, i) - NIIFC(j, i)/tIFC(j, i));
    NIstorage(j + 1, i) = NIstorage(j, i) + h*(NIIFC(j, i)/tIFC(j, i) - Ntburn(j, i)/TBE);
    end
    time(j) = t;
    t = t + h; %0.1
    j = j + 1;
    end
    j = 1;
    t = t0; %0
    i = i + 1;
end

%Graphs

if a == 1 && b == 0 && c == 0 && d == 0 && e == 0 %Ntburn randomized only
    figure(7)
    histogram(Istorage(steps, :), 'Normalization', 'pdf')
    hold on
    x4 = min(Istorage(steps, :)):(max(Istorage(steps, :)) - min(Istorage(steps, :)))/1000:max(Istorage(steps, :));
    f4 = zeros(1, length(x4));
    mu4 = mean(Istorage(steps, :));
    sigma4 = std(Istorage(steps, :));
    i = 1;
    while i <= length(x4)
        f4(i) = exp(-(x4(i) - mu4)^2/(2*sigma4^2))/(sigma4*sqrt(2*pi));
        i = i + 1;
    end
    plot(x4,f4)

elseif a == 0 && b == 1 && c == 0 && d == 0 && e == 0 %TBR randomized only
    figure(1)
    histogram(Istorage(steps, :), 'Normalization', 'pdf')
    hold on
    x4 = min(Istorage(steps, :)):(max(Istorage(steps, :)) - min(Istorage(steps, :)))/1000:max(Istorage(steps, :));
    f4 = zeros(1, length(x4));
    mu4 = mean(Istorage(steps, :));
    sigma4 = std(Istorage(steps, :));
    i = 1;
    while i <= length(x4)
        f4(i) = exp(-(x4(i) - mu4)^2/(2*sigma4^2))/(sigma4*sqrt(2*pi));
        i = i + 1;
    end
    plot(x4,f4)
    xlabel('Tritium in storage (kg)')
    ylabel('Probability density')
    %set(gcf, 'PaperPosition', [0 0 8 6]);
    %print('-dpng', 'Storage_TBR.png');

    figure(2)
    hold on
    i = 1;
    while i <= samples
        plot(time, Istorage(:, i))
        plot(time, NIstorage(:, i), '--')
        i = i + 1;
    end
    xlabel('time (h)')
    ylabel('Tritium in storage (kg)')
    legend('Exact', 'Euler')
    %set(gcf, 'PaperPosition', [0 0 8 6]);
    %print('-dpng', 'Storage_time.png');

elseif a == 0 && b == 0 && c == 1 && d == 0 && e == 0 %Istartup randomized only
    figure(7)
    histogram(Istorage, 'Normalization', 'pdf')
    hold on
    x4 = min(Istorage):(max(Istorage) - min(Istorage))/1000:max(Istorage);
    f4 = zeros(1, length(x4));
    mu4 = mean(Istorage);
    sigma4 = std(Istorage);
    i = 1;
    while i <= length(x4)
        f4(i) = exp(-(x4(i) - mu4)^2/(2*sigma4^2))/(sigma4*sqrt(2*pi));
        i = i + 1;
    end
    plot(x4,f4)
    xlabel('Tritium in storage (kg) after one cycle with only initial tritium randomised')
    ylabel('Probability density')
    %set(gcf, 'PaperPosition', [0 0 8 6]);
    %print('-dpng', 'Random_initial.png');
elseif a == 0 && b == 0 && c == 0 && d == 1 && e == 0 %tIFC randomized only
    figure(7)
    histogram(Istorage, 'Normalization', 'pdf')
    hold on
    x4 = min(Istorage):(max(Istorage) - min(Istorage))/1000:max(Istorage);
    f4 = zeros(1, length(x4));
    mu4 = mean(Istorage);
    sigma4 = std(Istorage);
    i = 1;
    while i <= length(x4)
        f4(i) = exp(-(x4(i) - mu4)^2/(2*sigma4^2))/(sigma4*sqrt(2*pi));
        i = i + 1;
    end
    plot(x4,f4)
elseif a == 0 && b == 0 && c == 0 && d == 0 && e == 1 %tOFC randomized only
    figure(7)
    histogram(Istorage, 'Normalization', 'pdf')
    hold on
    x4 = min(Istorage):(max(Istorage) - min(Istorage))/1000:max(Istorage);
    f4 = zeros(1, length(x4));
    mu4 = mean(Istorage);
    sigma4 = std(Istorage);
    i = 1;
    while i <= length(x4)
        f4(i) = exp(-(x4(i) - mu4)^2/(2*sigma4^2))/(sigma4*sqrt(2*pi));
        i = i + 1;
    end
    plot(x4,f4)
    samplemean = sum(Istorage)/samples;
    samplesd = sqrt(sum(Istorage.^2)/samples - samplemean^2);
    f5 = zeros(1, length(x4));
    i = 1;
    while i <= length(x4)
        f5(i) = exp(-(x4(i) - samplemean)^2/(2*samplesd^2))/(samplesd*sqrt(2*pi));
        i = i + 1;
    end
    plot(x4,f5)
elseif a == 1 && b == 1 && c == 1 && d == 1 && e == 1 %All parameters randomized
    figure(7)
    histogram(Istorage, 'Normalization', 'pdf')
    hold on
    x4 = min(Istorage):(max(Istorage) - min(Istorage))/1000:max(Istorage);
    f4 = zeros(1, length(x4));
    mu4 = mean(Istorage);
    sigma4 = std(Istorage);
    i = 1;
    while i <= length(x4)
        f4(i) = exp(-(x4(i) - mu4)^2/(2*sigma4^2))/(sigma4*sqrt(2*pi));
        i = i + 1;
    end
    plot(x4,f4)
    xlabel('Tritium in storage (kg) after one cycle with all parameters randomised')
    ylabel('Probability density')
    %set(gcf, 'PaperPosition', [0 0 8 6]);
    %print('-dpng', 'Random_all.png');
end