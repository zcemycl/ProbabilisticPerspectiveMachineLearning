clear all; close all; clc;

step_y = 0.08; step_x = 0.15; % step_x cannot be greater than 0.2 or == 0.1
d_two_legs = 0.2;
height = 0.0005; L = 0.1;
n = 20; new_ref = 0;
M = 0.1;
RInertia = M*L^2/3;

step_x_step = 0.14;

nfrb = floor(0.5/step_x);


% initialization
lineObj = animInit(height);

% --------------------------------------------------------------- %
% Motion trajectory
% --------------------------------------------------------------- %
% rear flat
xir = d_two_legs; xfr = xir - step_x; 

[phiIr, phiMr, phiFr, phiIMr, phiMFr] = angle_step(xir, xfr,...
                                        0, 0, step_y, n, 'r');
% front flat
xif = -xfr; xff = xif - step_x; 
[phiIf, phiMf, phiFf, phiIMf, phiMFf] = angle_step(xif, xff,...
                                        0, 0, step_y, n, 'f');

% front step                                    
xifs = step_x - d_two_legs; xffs = xifs - step_x_step; 
[phiIfs, phiMfs, phiFfs, phiIMfs, phiMFfs] = angle_step(xifs, xffs,...
                                        0, height, step_y, n, 'f');    
% rear step
xirs = -xffs; xfrs = 0.05; 
[phiIrs, phiMrs, phiFrs, phiIMrs, phiMFrs] = angle_step(xirs, xfrs,...
                                        -height, 0, step_y, n, 'r');
% --------------------------------------------------------------- %
% Energy & Torque
% --------------------------------------------------------------- %
% rear flat
T1r1 = []; T2r1 = []; T3r1 = [];
Er = 0;
PEr = 0; 
[r1T, r2T, r3T, r4T] = transformation(phiIr, L, 'r');
MID = midpoint_link(r1T, r2T, r3T, r4T, 'r');
for i =  1:n
    Phi = phiIr + phiIMr*i;
    [T, energy] = total_energy(Phi, phiIMr, 'r');
    Er = Er + energy;
    [P, M] = potential(MID, Phi, L, 'r');
    PEr = PEr + P;
    MID = M;
    
    T1r1(i) = T(1);
    T2r1(i) = T(2);
    T3r1(i) = T(3);
end


T1r2 = []; T2r2 = []; T3r2 = [];
for i =  1:n
    Phi = phiMr + phiMFr*i;
    [T, energy] = total_energy(Phi, phiMFr, 'r');
    Er = Er + energy;
    [P, M] = potential(MID, Phi, L, 'r');
    PEr = PEr + P;
    MID = M;
    
    T1r2(i) = T(1);
    T2r2(i) = T(2);
    T3r2(i) = T(3);
end


% front flat
T1f1 = []; T2f1 = []; T3f1 = [];
Ef = 0;
PEf = 0; 
[r1T, r2T, r3T, r4T] = transformation(phiIf, L, 'f');
MID = midpoint_link(r1T, r2T, r3T, r4T, 'f');
for i =  1:n
    Phi = phiIf + phiIMf*i;
    [T, energy] = total_energy(Phi, phiIMf, 'f');
    Ef = Ef + energy;
    [P, M] = potential(MID, Phi, L, 'f');
    PEf = PEf + P;
    MID = M;
    
    T1f1(i) = T(1);
    T2f1(i) = T(2);
    T3f1(i) = T(3);
end

T1f2 = []; T2f2 = []; T3f2 = [];
for i =  1:n
    Phi = phiMf + phiMFf*i;
    [T, energy] = total_energy(Phi, phiMFf, 'f');
    Ef = Ef + energy;
    [P, M] = potential(MID, Phi, L, 'f');
    PEf = PEf + P;
    MID = M;
    
    T1f2(i) = T(1);
    T2f2(i) = T(2);
    T3f2(i) = T(3);
end

% front step
T1f1s = []; T2f1s = []; T3f1s = [];
Efs = 0;
PEfs = 0; 
[r1T, r2T, r3T, r4T] = transformation(phiIfs, L, 'f');
MID = midpoint_link(r1T, r2T, r3T, r4T, 'f');
for i =  1:n
    Phi = phiIfs + phiIMfs*i;
    [T, energy] = total_energy(Phi, phiIMfs, 'f');
    Efs = Efs + energy;
    [P, M] = potential(MID, Phi, L, 'f');
    PEfs = PEfs + P;
    MID = M;
    T1f1s(i) = T(1);
    T2f1s(i) = T(2);
    T3f1s(i) = T(3);
end

T1f2s = []; T2f2s = []; T3f2s = [];
for i =  1:n
    Phi = phiMfs + phiMFfs*i;
    [T, energy] = total_energy(Phi, phiMFfs, 'f');
    Efs = Efs + energy;
    [P, M] = potential(MID, Phi, L, 'f');
    PEfs = PEfs + P;
    MID = M;
    
    T1f2s(i) = T(1);
    T2f2s(i) = T(2);
    T3f2s(i) = T(3);
end

% rear step
T1r1s = []; T2r1s = []; T3r1s = [];
Ers = 0;
PErs = 0; 
[r1T, r2T, r3T, r4T] = transformation(phiIrs, L, 'r');
MID = midpoint_link(r1T, r2T, r3T, r4T, 'r');
for i =  1:n
    Phi = phiIrs + phiIMrs*i;
    [T, energy] = total_energy(Phi, phiIMrs, 'r');
    Ers = Ers + energy;
    [P, M] = potential(MID, Phi, L, 'r');
    PErs = PErs + P;
    MID = M;
    
    T1r1s(i) = T(1);
    T2r1s(i) = T(2);
    T3r1s(i) = T(3);
end

T1r2s = []; T2r2s = []; T3r2s = [];
for i =  1:n
    Phi = phiMrs + phiMFrs*i;
    [T, energy] = total_energy(Phi, phiMFrs, 'r');
    Ers = Ers + energy;
    [P, M] = potential(MID, Phi, L, 'r');
    PErs = PErs + P;
    MID = M;
    
    T1r2s(i) = T(1);
    T2r2s(i) = T(2);
    T3r2s(i) = T(3);
end
t1r1 = time(phiIMr, T1r1, 'r');
t1r2 = time(phiMFr, T1r2, 'r');
t1f1 = time(phiIMf, T3f1, 'f');
t1f2 = time(phiMFf, T3f2, 'f');
t1r1s = time(phiIMrs, T1r1s, 'r');
t1r2s = time(phiMFrs, T1r2s, 'r');
t1f1s = time(phiIMfs, T3f1s, 'f');
t1f2s = time(phiMFfs, T3f2s, 'f');

maxT1 = max([T1r1, T1r2, T1f1, T1f2, T1r1s, T1r2s, T1f1s, T1f2s]);
maxT2 = max([T2r1, T2r2, T2f1, T2f2, T2r1s, T2r2s, T2f1s, T2f2s]);
maxT3 = max([T3r1, T3r2, T3f1, T3f2, T3r1s, T3r2s, T3f1s, T3f2s]);
ENERGY = Ef*(nfrb+3) + Efs + Er*(nfrb+1+3) + Ers;
Gravitywork = PEf*(nfrb+3) + PEfs +  PEr*(nfrb+1+3) + PErs;

disp('Time'); disp((t1r1+t1r2)*(nfrb+1+3)+(t1f1+t1f2)*(nfrb+3)+t1r1s+t1r2s+t1f1s+t1f2s);
disp('Energy Efficiency'); disp(1-Gravitywork/ENERGY);
disp('min T1,T2,T3'); disp(maxT1); disp(maxT2); disp(maxT3);

% --------------------------------------------------------------- %
% Motion Animation
% --------------------------------------------------------------- %
h = gcf;
% Capture the plot as an image 
frame = getframe(h); 
im = frame2im(frame); 
[imind,cm] = rgb2ind(im,256); 
% Write to the GIF File 
imwrite(imind,cm,'WalkingRobot.gif','gif', 'Loopcount',inf); 
                                    
for j = 1:nfrb
new_ref = motion(phiIr, phiMr, phiIMr, phiMFr, 'r', L, lineObj, new_ref, 0, n);
new_ref = motion(phiIf, phiMf, phiIMf, phiMFf, 'f', L, lineObj, new_ref, 0, n);
end
new_ref = motion(phiIr, phiMr, phiIMr, phiMFr, 'r', L, lineObj, new_ref, 0, n);
new_ref = motion(phiIfs, phiMfs, phiIMfs, phiMFfs, 'f', L, lineObj, new_ref, 0, n);
new_ref = motion(phiIrs, phiMrs, phiIMrs, phiMFrs, 'r', L, lineObj, new_ref, height, n);
for k = 1:3
new_ref = motion(phiIf, phiMf, phiIMf, phiMFf, 'f', L, lineObj, new_ref, height, n);
new_ref = motion(phiIr, phiMr, phiIMr, phiMFr, 'r', L, lineObj, new_ref, height, n);
end
new_ref = motion(phiIf, phiMf, phiIMf, phiMFf, 'f', L, lineObj, new_ref, height, n);



% ------------------------------------------------5-------- %
% Functions
% --------------------------------------------------------------- %
function lineObj = animInit(height)
hold on
axis equal
axis([-1.2 0.3 0 0.25])
plot([-1.2 -0.5 -0.5],[height height 0],...
         'k','LineWidth',3);

%link line objects
lineObj.h1T = line(0,0,'color','k','LineWidth',2);
lineObj.h2T = line(0,0,'color','k','LineWidth',2);
lineObj.h3T = line(0,0,'color','k','LineWidth',2);
lineObj.h4T = line(0,0,'color','k','LineWidth',2);
end 

function new_ref = motion(phiI, phiM, phiIM, phiMF, mode, L, lineObj, xref, yref, n)
if mode == 'r'
    for i = 1:n
        Phi = phiI + phiIM*i;
        [r1T, r2T, r3T, r4T] = transformation(Phi, L, 'r');
        animation(r1T, r2T, r3T, r4T, xref, yref, lineObj, 'r');
        gifGenerator()
    end
    
    for i = 1:n
        Phi2 = phiM + phiMF*i;
        [r1T, r2T, r3T, r4T] = transformation(Phi2, L, 'r');
        [new_ref] = animation(r1T, r2T, r3T, r4T, xref, yref, lineObj, 'r');
        gifGenerator()
    end
elseif mode == 'f'
    for i = 1:n
        Phi = phiI + phiIM*i;
        [r1T, r2T, r3T, r4T] = transformation(Phi, L, 'f');
        animation(r1T, r2T, r3T, r4T, xref, yref, lineObj, 'f');
        gifGenerator()
    end
    
    for i = 1:n
        Phi2 = phiM + phiMF*i;
        [r1T, r2T, r3T, r4T] = transformation(Phi2, L, 'f');
        [new_ref] = animation(r1T, r2T, r3T, r4T, xref, yref, lineObj, 'f');
        gifGenerator()
    end
end
end

function [r1T, r2T, r3T, r4T] = transformation(Phi, L, mode)
phi1_T = Phi(1); phi2_T = Phi(2); phi3_T = Phi(3);
A10T = [1 , 0 ,  0; ...
        0 , 1 , L; ...
        0 , 0 , 1 ];
if mode == 'r'
    A21T = [cos(phi1_T) , sin(phi1_T) , L*sin(phi1_T) ; ...
           -sin(phi1_T) ,  cos(phi1_T) , L*cos(phi1_T); ...
            0        , 0          , 1  ];

    A32T = [cos(phi2_T) , sin(phi2_T) , L*sin(phi2_T); ...
           -sin(phi2_T) ,  cos(phi2_T) , L*cos(phi2_T); ...
           0         , 0         , 1 ];
    A43T = [cos(phi3_T) , sin(phi3_T) , L*sin(phi3_T); ...
           -sin(phi3_T) ,  cos(phi3_T) , L*cos(phi3_T); ...
            0         , 0         , 1 ];
elseif mode == 'f'
    A21T = [cos(phi1_T) , -sin(phi1_T) , -L*sin(phi1_T) ; ...
            sin(phi1_T) ,  cos(phi1_T) , L*cos(phi1_T); ...
            0        , 0          , 1  ];

    A32T = [cos(phi2_T) , -sin(phi2_T) , -L*sin(phi2_T); ...
            sin(phi2_T) ,  cos(phi2_T) , L*cos(phi2_T); ...
            0         , 0         , 1 ];
    A43T = [cos(phi3_T) , -sin(phi3_T) , -L*sin(phi3_T); ...
            sin(phi3_T) ,  cos(phi3_T) , L*cos(phi3_T); ...
            0         , 0         , 1 ];
end
if mode == 'r'
    r1T = A10T(:,3);   %position link 1
    r1T = r1T(1:2);
    r2T = A10T*A21T;   %position link 2
    r2T = r2T(:,3);
    r2T = r2T(1:2);
    r3T = A10T*A21T*A32T;
    r3T = r3T(:,3);
    r3T = r3T(1:2);
    r4T = A10T*A21T*A32T*A43T;
    r4T = r4T(:,3);
    r4T = r4T(1:2);
elseif mode == 'f'
    r1T = A10T*A43T*A32T*A21T;   %position link 1
    r1T = r1T(:,3);
    r1T = r1T(1:2);
    r2T = A10T*A43T*A32T;   %position link 2
    r2T = r2T(:,3);
    r2T = r2T(1:2);
    r3T = A10T*A43T;
    r3T = r3T(:,3);
    r3T = r3T(1:2);
    r4T = A10T;
    r4T = r4T(:,3);
    r4T = r4T(1:2);
end
end

function [new_ref] = animation(r1T, r2T, r3T, r4T, x_ref, y_ref, lineObj, mode)
if mode == 'r'
    %set target position and pause
    set(lineObj.h1T,'xdata',[x_ref x_ref+r1T(1)],'ydata',[y_ref y_ref+r1T(2)])
    set(lineObj.h2T,'xdata',[x_ref+r1T(1) x_ref+r2T(1)],'ydata',[y_ref+r1T(2) y_ref+r2T(2)])
    set(lineObj.h3T,'xdata',[x_ref+r2T(1) x_ref+r3T(1)],'ydata',[y_ref+r2T(2) y_ref+r3T(2)])
    set(lineObj.h4T,'xdata',[x_ref+r3T(1) x_ref+r4T(1)],'ydata',[y_ref+r3T(2) y_ref+r4T(2)])
    drawnow;

    new_ref = x_ref+r4T(1);
elseif mode == 'f'
    %set target position and pause
    set(lineObj.h1T,'xdata',[x_ref+r1T(1) x_ref+r2T(1)],'ydata',[y_ref+r1T(2) y_ref+r2T(2)])
    set(lineObj.h2T,'xdata',[x_ref+r2T(1) x_ref+r3T(1)],'ydata',[y_ref+r2T(2) y_ref+r3T(2)])
    set(lineObj.h3T,'xdata',[x_ref+r4T(1) x_ref+r3T(1)],'ydata',[y_ref+r4T(2) y_ref+r3T(2)])
    set(lineObj.h4T,'xdata',[x_ref x_ref+r4T(1)],'ydata',[y_ref y_ref+r4T(2)])
    drawnow;

    new_ref = x_ref+r1T(1);
end


end

function [phiI, phiM, phiF, phiIM, phiMF] = angle_step(xi, xf, yi, yf, step_y, n, mode)
    phiI = pure_angle(xi, yi, mode);
    phiM = pure_angle(0.5*(xi+xf), 0.5*(yi+yf+2*step_y), mode);
    phiF = pure_angle(xf, yf, mode);
    
    phiIM = (phiM - phiI)/n;
    phiMF = (phiF - phiM)/n;
end

function Phi = pure_angle(x,y,mode)
L = 0.1; 
phi2 = acos((x^2 + y^2)/2/L^2 - 1);
if mode == 'f'
    phi1 = asin(0.5*(-x/L + sin(phi2)*y/L/(1+cos(phi2))));
    phi3 = pi - phi1 - phi2;
elseif mode == 'r'
    phi3 = asin(0.5*(x/L + sin(phi2)*((y-L)/L +1)/(cos(phi2)+1)));
    phi1 = pi - phi2 - phi3; 
end
Phi = [phi1, phi2, phi3];

end 


function [T, energy] = total_energy(Phi, dPhi, mode)                                   
phi1 = Phi(1); phi2 = Phi(2); phi3 = Phi(3);
L = 0.1; g = 9.81; M = 0.4;
if mode == 'r'
    tau1 = L*M*g*(sin(phi1)+sin(phi1+phi2)+sin(phi1+phi2+phi3));
    tau2 = L*M*g*(sin(phi1+phi2)+sin(phi1+phi2+phi3));
    tau3 = L*M*g*(sin(phi1+phi2+phi3));
elseif mode == 'f'
    tau1 = L*M*g*(sin(phi1+phi2+phi3));
    tau2 = L*M*g*(sin(phi2+phi3)+sin(phi1+phi2+phi3));
    tau3 = L*M*g*(sin(phi3)+sin(phi2+phi3)+sin(phi1+phi2+phi3));
end
T = [tau1, tau2, tau3];
energy = abs(T)*abs(dPhi)';
end

function [PE, MID] = potential(M, Phi, L,mode)
[r1T, r2T, r3T, r4T] = transformation(Phi, L, mode);
mid1 = M(1); mid2 = M(2); mid3 = M(3); mid4 = M(4);
if mode == 'r'
    PE1 = abs((r1T(2))*0.5 - mid1)*0.1*9.81;
    PE2 = abs((r1T(2)+r2T(2))*0.5 - mid2)*0.1*9.81;
    PE3 = abs((r2T(2)+r3T(2))*0.5 - mid3)*0.1*9.81;
    PE4 = abs((r3T(2)+r4T(2))*0.5 - mid4)*0.1*9.81;
elseif mode == 'f'
    PE1 = abs((r1T(2)+r2T(2))*0.5 - mid1)*0.1*9.81;
    PE2 = abs((r2T(2)+r3T(2))*0.5 - mid2)*0.1*9.81;
    PE3 = abs((r4T(2)+r3T(2))*0.5 - mid3)*0.1*9.81;
    PE4 = abs(r4T(2)*0.5 - mid4)*0.1*9.81;
end
MID = midpoint_link(r1T, r2T, r3T, r4T, mode);
PE = PE1+PE2+PE3+PE4;
end

function MID = midpoint_link(r1T, r2T, r3T, r4T, mode)
if mode == 'r'
    mid1 = (r1T(2))*0.5;
    mid2 = (r1T(2)+r2T(2))*0.5;
    mid3 = (r2T(2)+r3T(2))*0.5;
    mid4 = (r3T(2)+r4T(2))*0.5;
elseif mode == 'f'
    mid1 = (r1T(2)+r2T(2))*0.5;
    mid2 = (r2T(2)+r3T(2))*0.5;
    mid3 = (r4T(2)+r3T(2))*0.5;
    mid4 = r4T(2)*0.5;
end
MID = [mid1, mid2, mid3, mid4];
end

function t = time(Phi, Torque, mode)
    dim = size(Torque);
    a_profile = 0.001; % 0< <0.5
    I = 0.1^3/3;
    if mode == 'r'
        t = sqrt(I*sum(abs(ones(dim)*Phi(1)./Torque))/a_profile/(1-a_profile));
    elseif mode == 'f'
        t = sqrt(I*sum(abs(ones(dim)*Phi(3)./Torque))/a_profile/(1-a_profile));
    end
end

function gifGenerator()
h = gcf;
% Capture the plot as an image 
frame = getframe(h); 
im = frame2im(frame); 
[imind,cm] = rgb2ind(im,256); 
% Write to the GIF File 
imwrite(imind,cm,'WalkingRobot.gif','gif','WriteMode','append'); 
end