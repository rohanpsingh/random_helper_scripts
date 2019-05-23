tru_M = csvread('/home/rohan/rohan_m15x/results/true_pose.csv');
est2_M = csvread('/home/rohan/rohan_m15x/results/est_pose.csv');
%est3_M = csvread('/home/rohan/rohan_m15x/results/orb_slam2_odom.csv');

tru_x = tru_M(:,1)*100;
tru_y = tru_M(:,2)*100;
tru_z = tru_M(:,3)*100;
tru_r = tru_M(:,4);
tru_p = tru_M(:,5);
tru_w = tru_M(:,6);
tru_time = tru_M(:,7);
tru_time = tru_time - tru_time(1,1);

est2_x = est2_M(:,1)*100;   
est2_y = est2_M(:,2)*100;
est2_z = est2_M(:,3)*100;
est2_r = est2_M(:,4);
est2_p = est2_M(:,5);
est2_w = est2_M(:,6);
est2_time = est2_M(:,7);
est2_time = est2_time - est2_time(1,1);

% est3_x = est3_M(:,1)*100;
% est3_y = est3_M(:,2)*100;
% est3_z = est3_M(:,3)*100;
% est3_r = est3_M(:,4);
% est3_p = est3_M(:,5);
% est3_w = est3_M(:,6);
% est3_time = est3_M(:,7);
% est3_time = est3_time - est3_time(1,1);
 
 

window = [800 800 2000 1000];
figure('Position',window);

subplot(2,3,1);
plot(est2_time,est2_x,'r-o',tru_time,tru_x,'g');
title('x / time')

subplot(2,3,2);
plot(est2_time,est2_y,'r-o',tru_time,tru_y,'g');
title('y / time')

subplot(2,3,3);
plot(est2_time,est2_z,'r-o',tru_time,tru_z,'g');
title('z / time')

subplot(2,3,4);
plot(est2_time,est2_r,'r-o',tru_time,tru_r,'g');
title('euler0 / time')

subplot(2,3,5);
plot(est2_time,est2_p,'r-o',tru_time,tru_p,'g');
title('euler1 / time')

subplot(2,3,6);
plot(est2_time,est2_w,'r-o',tru_time,tru_w,'g');
title('euler2 / time')

% figure();
% plot(est_time,theta1,est_time,theta2,est_time,theta3);
% title('error in rpy / time');
