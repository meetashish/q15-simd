% author Ashish Kumar Meshram

clear;clc;clf;
xfp;
yfp;
zfp_sisd;
zfp_SISD1;
zfp_simd_sse;
zfp_SIMD1_SSE;
zfp_simd_avx256;
zfp_SIMD1_AVX256;

load('xfp.mat');
xfp = iqsamples;

load('yfp.mat');
yfp = iqsamples;

load('zfp_sisd.mat');
zfp_sisd = iqsamples;

load('zfp_SISD1.mat');
zfp_SISD1 = iqsamples;

load('zfp_simd_sse.mat');
zfp_simd_sse = iqsamples;

load('zfp_SIMD1_SSE.mat');
zfp_SIMD1_SSE = iqsamples;

load('zfp_simd_avx256.mat');
zfp_simd_avx256 = iqsamples;

load('zfp_SIMD1_AVX256.mat');
zfp_SIMD1_AVX256 = iqsamples;

zfp = xfp.*yfp;

%% Error Analysis
err_spfp_q15 = abs(zfp_sisd - zfp_SISD1);
err_sse_q15 = abs(zfp_simd_sse - zfp_SIMD1_SSE);
err_avx256_q15 = abs(zfp_simd_avx256 - zfp_SIMD1_AVX256);

%% Plot Results
subplot(311);
plot(0:numel(zfp) - 1, real(zfp), '-d', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(0:numel(zfp) - 1, real(zfp_sisd), '-d', 'LineWidth', 2, 'MarkerSize', 8);
plot(0:numel(zfp) - 1, real(zfp_SISD1), '-*', 'LineWidth', 2, 'MarkerSize', 9);
plot(0:numel(zfp) - 1, real(zfp_simd_sse), '-s', 'LineWidth', 2, 'MarkerSize', 10);
plot(0:numel(zfp) - 1, real(zfp_SIMD1_SSE), '-s', 'LineWidth', 2, 'MarkerSize', 10);
plot(0:numel(zfp) - 1, real(zfp_simd_avx256), '-s', 'LineWidth', 2, 'MarkerSize', 10);
plot(0:numel(zfp) - 1, real(zfp_SIMD1_AVX256), '-s', 'LineWidth', 2, 'MarkerSize', 10);
hold off;
grid on;
xlim([0, numel(zfp) - 1]);
xlabel('Number of I samples','Interpreter','latex');
ylabel('Inphase value','Interpreter','latex');
legend('MATLAB', 'C (SPFP)', 'C (Q15)', 'C (SPFP-SSE)', 'C (Q15-SSE)', 'C (SPFP-AVX256)', 'C (Q15-AVX256)', 'Interpreter','latex', 'Location', 'best');

subplot(312);
plot(0:numel(zfp) - 1, imag(zfp), '-d', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(0:numel(zfp) - 1, imag(zfp_sisd), '-d', 'LineWidth', 2, 'MarkerSize', 8);
plot(0:numel(zfp) - 1, imag(zfp_SISD1), '-*', 'LineWidth', 2, 'MarkerSize', 9);
plot(0:numel(zfp) - 1, imag(zfp_simd_sse), '-s', 'LineWidth', 2, 'MarkerSize', 10);
plot(0:numel(zfp) - 1, imag(zfp_SIMD1_SSE), '-s', 'LineWidth', 2, 'MarkerSize', 10);
plot(0:numel(zfp) - 1, imag(zfp_simd_avx256), '-s', 'LineWidth', 2, 'MarkerSize', 10);
plot(0:numel(zfp) - 1, imag(zfp_SIMD1_AVX256), '-s', 'LineWidth', 2, 'MarkerSize', 10);
hold off;
grid on;
xlim([0, numel(zfp) - 1]);
xlabel('Number of Q samples','Interpreter','latex');
ylabel('Quadrature value','Interpreter','latex');
legend('MATLAB', 'C (SPFP)', 'C (Q15)', 'C (SPFP-SSE)', 'C (Q15-SSE)', 'C (SPFP-AVX256)', 'C (Q15-AVX256)', 'Interpreter','latex', 'Location', 'best');

subplot(313);
plot(0:numel(zfp) - 1, err_spfp_q15, 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(0:numel(zfp) - 1, err_sse_q15, 'LineWidth', 2, 'MarkerSize', 8);
plot(0:numel(zfp) - 1, err_avx256_q15, 'LineWidth', 2, 'MarkerSize', 8);
hold off;
grid on;
xlim([0, numel(zfp) - 1]);
xlabel('Number of IQ samples','Interpreter','latex');
ylabel('Error','Interpreter','latex');
legend('SPFP to Q15', 'SPFP SSE to Q15-SSE', 'SPFP AVX256 to Q15-AVX256', 'Interpreter','latex', 'Location', 'best');