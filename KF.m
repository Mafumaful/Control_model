%% System Model and Measurement Model
% the model is free fall model

% set simulate time
dt = 0.01;
t = 0:dt:2;
N = length(t);

% set the attack mode
attack = true;

attack_signal = zeros(3,N);
if attack==true
    for k=1:N
        attack_signal(:,k)=[5*k*dt;5*k*dt;5*k*dt];
    end
end

% state-related var
X_real = zeros(9, N);
X_filtered = zeros(9, N);
error=zeros(3,N); % it represents local innovation

% suppose that we can measure the position
Z_measure = zeros(3, N);

pos = [0; 0; 0];
vel = [2; 3; 10];
acc = [0; 0; -9.8];
u = ones(1, N);

% kalman-related var
r_noise = 0.05;
q_noise = 0.01;

Q = q_noise * eye(9);
R = r_noise * eye(3);
P = eye(9);

W = sqrt(Q) * randn(9, N); % noise
V = sqrt(R) * randn(3, N); % noise
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

B = [
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    ];

% residule related
rsd = zeros(1,N);% plot the residual with time 
array = [];
J=10; % the length of dynamic window


%% Kalman Filter
for k = 2:N
    % update the physical model
    vel = vel + (acc + u(:, k)) * dt;
    pos = pos + vel * dt + 0.5 * acc * dt * dt;
    X_real(:, k) = [pos; vel; acc];

    % measure with noise + attack (if the mode is attack mode)
    Z_measure(:, k) = pos + V(:, k) + attack_signal(:,k);

    % kalman filter
    X_pre = F * X_filtered(:, k - 1) + B * u(k);
    P_pre = F * P * F' + Q;
    error(:,k) = Z_measure(:, k) - H * X_pre;
    K = P_pre * H' * inv(H * P_pre * H' + R);
    X_filtered(:, k) = X_pre + K * error(:,k);
    P = (eye(9) - K * H) * P_pre;

    % 
    residule = error(:,k)'*inv(H*P*H'+R)*error(:,k);
    array(end+1)=residule;
    % make sure the length of the window is 10
    if length(array)>10
        array(1)=[];
    end
    rsd(k)=sum(array);
end


%% Analysis
% plot the true value of it

figure('Name', 'Kalman Filter Simulation', 'NumberTitle', 'off');
plot3(X_real(1,:),X_real(2,:),X_real(3,:), ...
    X_filtered(1, :), X_filtered(2, :), X_filtered(3, :), 'r', ... % real
    Z_measure(1, :), Z_measure(2, :), Z_measure(3, :), 'b-o', MarkerSize = 3 ... % measure with noise
);
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
legend('true value','filtered value', 'value with measure noise');
grid on;
title("kalman filter in 3D");

figure('Name', 'Error Analysis', 'NumberTitle', 'off');
plot(t,error(1,:),'o',t,error(2,:),'o',t,error(3,:),'o');
legend('ex pos','ey pos', 'ez pos');
title("local innovation");
grid on;

figure('Name', 'residule', 'NumberTitle', 'off');
plot(t,rsd,'o');
legend('residual');
grid on;
