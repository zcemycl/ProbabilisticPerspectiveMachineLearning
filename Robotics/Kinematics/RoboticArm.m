clear all; close all; clc;
% create a function with inputs of current position
% mode (forefoot or backfoot)
% foot step height, foot step distance
L = 0.1;
% output current position 
% x = 0.15; y = 0.05; 
x = 0.12; y = 0; 
% x = 0.1; y = 0; 

phi2 = acos((x^2+(y-L)^2)/(2*L^2) + (y-L)/L -0.5);
phi2_d = phi2*180/pi;

phi3 = asin(0.5*(x/L + sin(phi2)*((y-L)/L +1)/(cos(phi2)+1)));
phi3_d = phi3*180/pi;


phi1_d = 180 - phi2_d - phi3_d;
phi1 = phi1_d*pi/180;

disp(['Angle 1:', num2str(phi1_d)])
disp(['Angle 2:', num2str(phi2_d)])
disp(['Angle 3:', num2str(phi3_d)]);

x_p = 0.1*(sin(phi1)+ sin(phi1+phi2)+ sin(phi1+phi2+phi3));
y_p = 0.1*(1+cos(phi1)+cos(phi1+phi2)+cos(phi1+phi2+phi3));


A10 = [1 , 0 ,  0; ...
   0 , 1 , L; ...
   0 , 0 , 1 ];

A21 = [cos(phi1) , sin(phi1) , L*sin(phi1) ; ...
       -sin(phi1) ,  cos(phi1) , L*cos(phi1); ...
        0        , 0          , 1  ];

A32 = [cos(phi2) , sin(phi2) , L*sin(phi2); ...
       -sin(phi2) ,  cos(phi2) , L*cos(phi2); ...
       0         , 0         , 1 ];
A43 = [cos(phi3) , sin(phi3) , L*sin(phi3); ...
       -sin(phi3) ,  cos(phi3) , L*cos(phi3); ...
        0         , 0         , 1 ];   
A54 = [ 0,1,L;...
       -1,0,0;... 
        0,0,1];

r11 = A10(:,3);   %position link 1
r11 = r11(1:2);
r22 = A10*A21;   %position link 2
r22 = r22(:,3);
r22 = r22(1:2);
r33 = A10*A21*A32;
r33 = r33(:,3);
r33 = r33(1:2);
r44 = A10*A21*A32*A43;
r44 = r44(:,3);
r44 = r44(1:2);
r55 = A10*A21*A32*A43*A54;
r55 = r55(:,3);
r55 = r55(1:2);


n = 100;
phi1_f = phi1/n;
phi2_f = phi2/n;
phi3_f = phi3/n;
hold on
axis equal
axis([-0.4 0.4 0 0.45])

lineObj.m1T = line(0,0,'color','r','LineWidth',5,'Marker','o');
set(lineObj.m1T,'xdata',x,'ydata',y)
%link line objects
lineObj.h1T = line(0,0,'color','k','LineWidth',2);
lineObj.h2T = line(0,0,'color','k','LineWidth',2);
lineObj.h3T = line(0,0,'color','k','LineWidth',2);
lineObj.h4T = line(0,0,'color','k','LineWidth',2);
%joint line objects
lineObj.d1T = line(0,0,'color','k','LineWidth',5,'Marker','o');
lineObj.d2T = line(0,0,'color','k','LineWidth',5,'Marker','o');
lineObj.d3T = line(0,0,'color','k','LineWidth',5,'Marker','o');
lineObj.d4T = line(0,0,'color','k','LineWidth',5,'Marker','o');


for i = 1:n
    
    phi1_T = phi1_f*i;
    phi2_T = phi2_f*i;
    phi3_T = phi3_f*i;
    A10T = [1 , 0 ,  0; ...
           0 , 1 , L; ...
           0 , 0 , 1 ];

    A21T = [cos(phi1_T) , sin(phi1_T) , L*sin(phi1_T) ; ...
           -sin(phi1_T) ,  cos(phi1_T) , L*cos(phi1_T); ...
            0        , 0          , 1  ];

    A32T = [cos(phi2_T) , sin(phi2_T) , L*sin(phi2_T); ...
           -sin(phi2_T) ,  cos(phi2_T) , L*cos(phi2_T); ...
           0         , 0         , 1 ];
    A43T = [cos(phi3_T) , sin(phi3_T) , L*sin(phi3_T); ...
           -sin(phi3_T) ,  cos(phi3_T) , L*cos(phi3_T); ...
            0         , 0         , 1 ];
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
    
    
    %set target position and pause
    set(lineObj.h1T,'xdata',[0 r1T(1)],'ydata',[0 r1T(2)])
    set(lineObj.h2T,'xdata',[r1T(1) r2T(1)],'ydata',[r1T(2) r2T(2)])
    set(lineObj.h3T,'xdata',[r2T(1) r3T(1)],'ydata',[r2T(2) r3T(2)])
    set(lineObj.h4T,'xdata',[r3T(1) r4T(1)],'ydata',[r3T(2) r4T(2)])
    set(lineObj.d1T,'xdata',r1T(1),'ydata',r1T(2))
    set(lineObj.d2T,'xdata',r2T(1),'ydata',r2T(2))
    set(lineObj.d3T,'xdata',r3T(1),'ydata',r3T(2))
    drawnow;

    h = gcf;
    % Capture the plot as an image 
    frame = getframe(h); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    if i == 1
      imwrite(imind,cm,'RoboticArm.gif','gif', 'Loopcount',inf); 
    else 
      imwrite(imind,cm,'RoboticArm.gif','gif','WriteMode','append'); 
    end 
    
end
