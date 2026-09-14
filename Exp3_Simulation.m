clc; clear; close all;
f = 3;                    % fringe frequency
N1=3;
N2=12;
c = 10;             % c denotes the scaling factor
k2=1:N2;
Sk_ideal=c*(cos(2*(k2-1)*pi/N2)+1);
Sk=round(Sk_ideal);  % The projection number
beta=Sk_ideal-Sk;    % rounding error β
Total_Projecion_number=sum(Sk);
c2=N2*c/2; c3=0;
for k2=1:N2
    c2=c2-cos(2*(k2-1)*pi/N2)*beta(k2);
    c3=c3+sin(2*(k2-1)*pi/N2)*beta(k2);
end
fprintf("Projection number during one exposure : %d \n Projeciton number for 12 patterns : ",Total_Projecion_number);
fprintf("%d,", Sk);
A=0.5;
B=0.4;
B_noise =0.1;    % The intensity of harmonic
order_noise=5;   % The order of harmonic
Height_img=500;
Width_img=500;
u=1:Width_img;
%%%%%%%%%%%% 一、Phase-shifting Superimposition (Internal phase-shifting method)
for k1=1:1:N1
    figure;I_k1=0;
    for k2=1:1:N2                 % Phase-shifting
        absolute_phase=2*pi*f*u/Width_img;
        phase=absolute_phase+2*(k1-1)*pi/N1+2*(k2-1)*pi/N2;
        noise=B_noise*cos(order_noise*phase);
        I_k1k2 = A + B*cos(phase)+noise;  % Projection patterns
        img=repmat(I_k1k2,Height_img,1);
%             imshow(img);
        subplot(4,4,k2);plot(I_k1k2,'LineWidth',1.5,'Color',[7/255,193/255,96/255]);
%         title(['Projection Pattern: I_{k1=',num2str(k1),', ','k2=',num2str(k2),'}'],'FontSize',7,'FontName','Times New Roman');
        xlabel('Pixel Position u','FontSize',10,'FontName','Times New Roman');
        ylabel('Intensity','FontSize',10,'FontName','Times New Roman'); set(gca,'ytick',0:0.5:1);
        I_k1=I_k1+I_k1k2*Sk(k2);   % Sumperimposition (achievd by one exposure of camera)
    end
    subplot(4,4,13:16); plot(I_k1,'LineWidth',1.5,'Color',[192/255,0/255,0/255]);  % Camera images
    set(gca,'xtick',0:100:500);
%     title(['Camera Image: I_{k1=',num2str(k1),'}'],'FontSize',7,'FontName','Times New Roman');
    xlabel('Pixel Position u','FontSize',10,'FontName','Times New Roman');
    ylabel('Intensity','FontSize',10,'FontName','Times New Roman');
end
%%%%%%%%%%%% 二、Traditional Projection
for k1=1:1:N1
    figure;I_k1=0;    set(gcf,'unit','centimeters','position',[10 10 13 13]);set(gca,'xtick',0:100:500);
    absolute_phase=2*pi*f*u/Width_img;
    phase=absolute_phase+2*(k1-1)*pi/N1;
    noise=B_noise*cos(order_noise*phase);
    I_k1 = A + B*cos(phase)+noise;  % Projection patterns
    subplot(2,1,1);plot(I_k1,'LineWidth',1.5,'Color',[7/255,193/255,96/255]);  set(gca,'ytick',0:0.5:1);
%     title(['Projection Pattern: I_{k1=',num2str(k1),',*}'],'FontSize',14,'FontName','Times New Roman');
    xlabel('Pixel Position u','FontSize',14,'FontName','Times New Roman');
    ylabel('Intensity','FontSize',14,'FontName','Times New Roman');
    I_k1=I_k1*Total_Projecion_number;   % Sumperimposition (achievd by one exposure of camera)
    subplot(2,1,2); plot(I_k1,'LineWidth',1.5,'Color',[192/255,0/255,0/255]);         % Camera images
    set(gca,'ytick',0:50:100); ylim([0,120]);
%     title(['Camera Image: I_{k1=',num2str(k1),'}'],'FontSize',14,'FontName','Times New Roman');
    xlabel('Pixel Position u','FontSize',15,'FontName','Times New Roman');
    ylabel('Intensity','FontSize',15,'FontName','Times New Roman');
end
%%

%%%数据读取
for id_method = 1:Method_number
    N=Step_name(id_method);
    for id_frequency=1:Frequency_number
        numerator=0;
        denominator=0;
        for k=1:N
            path=[root_path,Method_name{id_method},'\',num2str(id_frequency), '_', num2str(k),'.bmp'];
            Img=imread(path);
            Img=im2double(Img);
            % Img=filter2(fspecial('gaussian',5,2),Img);
            if Flag_Ours(id_method)==0
                numerator=numerator+Img*sin(2*(k-1)*pi/N);
                denominator=denominator+Img*cos(2*(k-1)*pi/N);
            else
                numerator=numerator-(c3*cos(2*(k-1)*pi/N)-c2*sin(2*(k-1)*pi/N))*Img; %%注意为减法
                denominator=denominator+(c2*cos(2*(k-1)*pi/N)+c3*sin(2*(k-1)*pi/N))*Img;
            end
        end
        phi(:,:,id_frequency)=-atan2(numerator,denominator)+pi;   %转化为 0-2*pi
    end
    %%%相位展开
    phl = phi(:,:,1);
    for i=1:Frequency_number-1
        phh = phi(:,:,i+1);
        kh = round((FrequencyRatio(i)*phl-phh)/(2*pi));
        phl = phh + kh*2*pi;
    end
    phi_unwrapped = phl;
    figure;imshow(phi_unwrapped,[]);title('相位图');
    path=[root_path,Method_name{id_method},'\','Phase.mat'];
    save(path, 'phi_unwrapped');
end