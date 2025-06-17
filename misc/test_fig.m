% Parameters
beta = 1e-6; % Confidence level
N = [2000, 4000, 8000]; % N1, N2, N3
k = linspace(0,N(2),200);

% Initialize arrays for risk bounds
epsL_values = zeros(size(k));
epsU_values = zeros(size(k));

% Calculate risk bounds for each k value
for i = 1:length(k)
    k_value = k(i);
    fprintf('k = %d\n', k_value);
    [epsL, epsU] = find_epsLU(k_value, N(2), beta);
    epsL_values(i) = epsL;
    epsU_values(i) = epsU;
end

% Plot the results
figure;
plot(k, epsL_values, '-', 'DisplayName', 'Lower Bound (epsL)');
hold on;
plot(k, epsU_values, '-', 'DisplayName', 'Upper Bound (epsU)');
xlabel('k');
ylabel('Risk Bounds');
title('Risk Bounds vs k');
legend('show');
ylim(0:1);
grid on;
hold off;