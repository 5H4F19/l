% 1. Unit Impulse Signal
t = -2:1:2;
y = [zeros(1,2), ones(1,1), zeros(1,2)];
figure(1);
subplot(2,2,1);
stem(t, y);
title('Unit Impulse');

% 2. Unit Step Signal
n = input('Enter n value for unit step signal: ');
t = 0:1:n-1;
y = ones(1, n);
subplot(2,2,2);
stem(t, y);
title('Unit Step');

% 3. Sine Wave
t = 0:0.01:2*pi;
y = sin(t);
subplot(2,2,3);
plot(t, y);
title('Sine Wave');

% 4. 10 Hz Signal
f = 10; a = 1; phi = 0;
t = 0:1/(f*100):1;
y = a * sin(2 * pi * f * t + phi);
subplot(2,2,4);
plot(t, y);
title('10 Hz Signal');

% 5. Unit Ramp Signal
n = input('Enter n value for unit ramp signal: ');
t = 0:n;
y = t;  % Ramp signal
figure(2);
subplot(2,1,1);
stem(t, y);
title('Unit Ramp');

% 6. Exponential Signal
n = input('Enter length of input sequence for exponential signal: ');
t = 0:n;
a = input('Enter a value for exponential signal: ');
y = exp(a * t);
subplot(2,1,2);
stem(t, y);
title('Exponential Signal');
xlabel('Time (n)');
ylabel('Amplitude');

% 7. Frequency Response of a System
b = input('Enter coefficients of x[n], x[n-1]: ');
a = input('Enter coefficients of y[n], y[n-1]: ');
N = input('Enter number of samples for frequency response: ');
[h, t] = freqz(b, a, N);
figure(3);
subplot(2,1,1);
plot(t, abs(h));  % Magnitude plot
title('Magnitude Response');
subplot(2,1,2);
stem(t, abs(h));  % Stem plot
title('Stem Plot of Frequency Response');

% 8. Impulse Response of a Given System
b = input('Enter coefficients of x[n], x[n-1], x[n-2]: ');
a = input('Enter coefficients of y[n], y[n-1], y[n-2]: ');
N = input('Enter number of samples for impulse response: ');
[h, t] = impz(b, a, N);
figure(4);
subplot(2,1,1);
plot(t, h);
title('Impulse Response');
subplot(2,1,2);
stem(t, h);
title('Stem Plot of Impulse Response');

% 9. DFT and IDFT of a Signal
w = [0:500]*pi/500;
j = 1i;
z = exp(-j * w);
x = 3 * (1 - 0.9 * z) ^ (-1);
a = abs(x);
b = angle(x) * 180 / pi;
figure(5);
subplot(2,1,1);
plot(w / pi, a);  % Magnitude
subplot(2,1,2);
plot(w / pi, b);  % Phase

% 10. DTFT (Discrete Time Fourier Transform) 
w = -4*pi:8*pi/511:4*pi;
num = [2, 1];
den = [1, -0.6];
h = freqz(num, den, w);
figure(6);
subplot(2,2,1);
plot(w / pi, real(h));
title('Real Part of H(e^{j\omega})');
subplot(2,2,2);
plot(w / pi, imag(h));
title('Imaginary Part of H(e^{j\omega})');
subplot(2,2,3);
plot(w / pi, abs(h));
title('Magnitude Spectrum |H(e^{j\omega})|');
subplot(2,2,4);
plot(w / pi, angle(h));
title('Phase Spectrum arg[H(e^{j\omega})]');

% 11. N-point DFT
N = input('Enter the value of N for N-point DFT: ');
x = input('Enter the sequence for DFT calculation: ');
n = 0:N-1;
k = 0:N-1;
WN = exp(-1i * 2 * pi / N);
nk = n' * k;
WNnK = WN .^ nk;
Xk = x * WNnK;
MagX = abs(Xk);  % Magnitude of DFT
PhaseX = angle(Xk) * 180 / pi;  % Phase of DFT
figure(7);
subplot(2,1,1);
plot(k, MagX);
title('Magnitude of DFT');
subplot(2,1,2);
plot(k, PhaseX);
title('Phase of DFT');
