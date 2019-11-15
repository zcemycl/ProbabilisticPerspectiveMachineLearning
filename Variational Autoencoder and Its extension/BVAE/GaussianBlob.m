R = .4; data = [];
h = figure;
count = 1;
filename = 'GaussBlob.gif';
for yc = 0:2*R/15:2*R
    for xc = 0:2*R/15:2*R
        gauss = generateImage(xc,yc,R);
        data = cat(3,data,gauss);
        imagesc( gauss, [0 1] );                        % display
        axis off; axis image;     % use gray colormap
        colormap gray
        drawnow;
        
        % Capture the plot as an image 
        frame = getframe(h); 
        im = frame2im(frame); 
        [imind,cm] = rgb2ind(im,256); 
        % Write to the GIF File 
        if count == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
        else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
        end 
        
        count = count+1;
    end
end


function gauss = generateImage(xc,yc,R)
% generate gaussian blobs dataset
imSize = 64;                           % image size: n X n
sigma = 1.5;
trim = .005;                             % trim off gaussian values smaller than this

% make linear ramp
X = 1:imSize;                           % X is a vector from 1 to imageSize
X0 = (X / imSize) - .5;                 % rescale X -> -.5 to .5
[Xm Ym] = meshgrid(X0, X0);
s = sigma / imSize;  

gauss = exp( -((((Xm-xc+R).^2)+((Ym-yc+R).^2)) ./ (s^2)) ); % formula for 2D gaussian
gauss(gauss < trim) = 0;                 % trim around edges (for 8-bit colour displays)
gauss(gauss >= trim) = 1;
end


