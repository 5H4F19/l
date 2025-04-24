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
x = [1, 2, 3, 4, 5]; % Example signal
X = fft(x); % DFT using MATLAB's fft function

% Plot the magnitude and phase of the DFT
subplot(2,1,1);
stem(0:length(X)-1, abs(X));
title('Magnitude of DFT');
xlabel('Frequency Index k');
ylabel('Magnitude');

subplot(2,1,2);
stem(0:length(X)-1, angle(X));
title('Phase of DFT');
xlabel('Frequency Index k');
ylabel('Phase (radians)');

IDFT
x_reconstructed = ifft(X); % Inverse DFT using MATLAB's ifft function

% Plot the original and reconstructed signal
subplot(2,1,1);
stem(0:length(x)-1, x);
title('Original Time Domain Signal');
xlabel('Time Index n');
ylabel('Amplitude');

subplot(2,1,2);
stem(0:length(x_reconstructed)-1, real(x_reconstructed)); % Take the real part of IDFT
title('Reconstructed Time Domain Signal (from IDFT)');
xlabel('Time Index n');
ylabel('Amplitude');



% 10. DTFT (Discrete Time Fourier Transform) 
x = [1, 2, 3, 4];  % Discrete-time signal
N = 1024;          % Number of frequency points

[X, w] = freqz(x, 1, N);  % Numerator = x, Denominator = 1 (no system)

% Plot
figure;
subplot(2,1,1);
plot(w/pi, abs(X));
title('Magnitude of DTFT (via freqz)');
xlabel('\omega / \pi');
ylabel('|X(\omega)|');
grid on;

subplot(2,1,2);
plot(w/pi, angle(X));
title('Phase of DTFT (via freqz)');
xlabel('\omega / \pi');
ylabel('Phase of X(\omega)');
grid on;


using FFT
x = [1, 2, 3, 4];
N = 1024;  % Zero-padding for higher resolution
X = fft(x, N);  % FFT is a sampled DTFT

w = linspace(0, 2*pi, N);  % Frequencies

% Plot
figure;
subplot(2,1,1);
plot(w/pi, abs(X));
title('Magnitude of DTFT (via fft)');
xlabel('\omega / \pi');
ylabel('|X(\omega)|');

subplot(2,1,2);
plot(w/pi, angle(X));
title('Phase of DTFT (via fft)');
xlabel('\omega / \pi');
ylabel('Phase of X(\omega)');
grid on;







% 11. N-point DFT
x = [1, 2, 3, 4];    % Your input signal
N = length(x);       % DFT size (same as signal length)

n = 0:N-1;           % Time index
k = n';              % Frequency index as column for matrix multiply

WN = exp(-1i * 2 * pi / N);  % DFT root of unity
WNnk = WN .^ (k * n);        % DFT matrix

X = WNnk * x.';              % DFT computation

% Plot magnitude and phase
figure;
subplot(2,1,1);
stem(0:N-1, abs(X));
title('Magnitude of DFT');
xlabel('Frequency index k');
ylabel('|X[k]|');

subplot(2,1,2);
stem(0:N-1, angle(X));
title('Phase of DFT');
xlabel('Frequency index k');
ylabel('∠X[k] (radians)');

