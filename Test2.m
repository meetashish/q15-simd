% author Ashish Kumar Meshram
clear;clc;clf;

MSE;
CPUtime;

load('mse.mat');
load('cpu_time.mat');

%% Plot Results
cat1 = categorical({'SPFP', 'SPFP-SSE', 'SPFP-AVX256', 'Q15', 'Q15-SSE', 'Q15-AVX256'});
cat1 = reordercats(cat1, {'SPFP', 'SPFP-SSE', 'SPFP-AVX256', 'Q15', 'Q15-SSE', 'Q15-AVX256'});
subplot(211);
bar(cat1, cpu_time/1000000);
grid on;
xlabel('Instruction categories','Interpreter','latex');
ylabel('CPU Time $(\mu s)$','Interpreter','latex');

subplot(212);
cat2 = categorical({'SPFP-SSE', 'SPFP-AVX256', 'Q15','Q15-SSE', 'Q15-AVX256'});
cat2 = reordercats(cat2, {'SPFP-SSE', 'SPFP-AVX256', 'Q15','Q15-SSE', 'Q15-AVX256'});
bar(cat2, mse);
grid on;
xlabel('Instruction categories','Interpreter','latex');
ylabel('MSE','Interpreter','latex');