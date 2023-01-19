%% System model and Observe
% free fall model

% set simulate time
dt = 0.01;
t = 0:dt:5;
N = length(t);

% set the attack mode
attack = false;
attack_signal = zeros(3, N);

if attack == true
    for k = 1:N
        if dt * k > 2
            attack_signal(:, k) = [ (k) * dt; (k) * dt; (k) * dt];
            % attack_signal(:, k) = 0.8.*[rand-0.5; rand-0.5; rand-0.5];
        end
    end

end

% measurement noise 
r_noise = 0.05;
R = r_noise*eye(3);
V = sqrt(R) * randn(3, N); % measurement noise

% state-related var
X_real = zeros(9, N);
X_filtered = zeros(9, N);
error = zeros(3, N); % it represents local innovation

% suppose that we can measure the position
Z_measure = zeros(3, N);

pos = [0; 0; 0];
vel = [2; 3; 10];
acc = [0; 0; -9.8];
u = ones(1, N);

% sliding mode observer
tilde_y = zeros(3,N);
tilde_x = zeros(9,N);

% transform matrix
F = [1 0 0 dt 0 0 0 0 0;
    0 1 0 0 dt 0 0 0 0;
    0 0 1 0 0 dt 0 0 0;
    0 0 0 1 0 0 dt 0 0;
    0 0 0 0 1 0 0 dt 0;
    0 0 0 0 0 1 0 0 dt;
    0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1; ];

H = [1 0 0 0 0 0 0 0 0;
    0 1 0 0 0 0 0 0 0;
    0 0 1 0 0 0 0 0 0; ];

B = [0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;];

v= [0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;];

%% sliding mode observer
for k = 2:N
    % update the physical model
    vel = vel + (acc + u(:, k)) * dt;
    pos = pos + vel * dt + 0.5 * acc * dt * dt;
    X_real(:, k) = [pos; vel; acc];

    % measure with noise + attack (if the mode is attack mode)
    Z_measure(:, k) = pos + V(:, k) + attack_signal(:, k);

    % observer
    tilde_dot_x = B*u(k) + F*tilde_x(:,k-1) - v;
    tilde_x(:,k) = tilde_dot_x + tilde_x(:,k-1);
    tilde_y(:,k) = H*tilde_x(:,k);
    error(:,k)=  Z_measure(:,k) - tilde_y(:,k);
    % v(1:3)=0.01.*error(:,k) + 0.01.*sign(error(:,k));

end


%% Analysis
figure('Name', 'Kalman Filter Simulation', 'NumberTitle', 'off');
plot3(X_real(1, :), X_real(2, :), X_real(3, :),'b', ...
    Z_measure(1, :), Z_measure(2, :), Z_measure(3, :), 'r-o', MarkerSize = 3 ... % measure with noise
);
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
legend('true value','value with measure noise');
grid on;
title("model in 3D");