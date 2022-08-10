clc;
clear all;close all;
ns=[32,8];
global nx ny nc b1c n3 r mk

kdata=[];
filenames={ 'multi_coil_lowres_speech.mat'}
load(filenames{1})
[~,name,~] = fileparts(filenames{1});
no_comp=8;
[k]= coil_compress_withpca(k,no_comp);
[nx,ny,nc,nt]=size(k);
n=nx*ny;
q=nt;
im = sqrt(nx*ny)*ifft2(k);
xm = mean(im,4);
csm = ismrm_estimate_csm_walsh_modified(xm);
b1=csm;
tmp = sqrt(sum(abs((b1)).^2,3));
b1c = div0(b1,tmp);
Xtrue = squeeze(sum(im.*repmat(conj(csm),[1 1 1 nt]),3));
radial=[4];
samp1 = goldencart(nx,ny,nt,radial);
mask=repmat(samp1,[1,1,1,nc]);
param.E=getE(b1c,nt,'samp',mask(:,:,:,1)~=0);
kdata=param.E*Xtrue;


global samp


[fid,msg] = fopen('Subspace_Tracking.txt','wt');
fprintf(fid, '%s & %s    \n','Subspace','AltGDMin MRI');
for ii=1:1:length(ns)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Subspace_Tracking (70,5)%%%%
    
    n3=nt/ns(ii);
    pp=n3;
    tic;
    Zhat=zeros(nx,ny,nt);
    for ww=1:1:ns(ii)
        param.E=getE(b1c,n3,'samp',kdata(:,:,(ww-1)*n3+1:ww*n3,1)~=0);
        samp=samp1(:,:,(ww-1)*n3+1:ww*n3);
        ys=kdata(:,:,(ww-1)*n3+1:ww*n3,:);
        T=5;
        if ww==1
            T=70;
        end
        [zbar_hat,flag,resNE,iter] = cgls_mean(@E_forw_for_mean,@E_back_for_mean, ys,0,1e-6,3);
        ybar_hat=E_forw_for_mean(zbar_hat);;
        
        yinter=ys-ybar_hat;
        mk=[];
        sum_mk=nnz(ys);
        for k=1:1:n3
            mk(k)=nnz(ys(:,:,k,:));
        end
        m=max(mk);
        if ww==1
            
            C_tilda=36;
            alpha=C_tilda*norm(yinter(:))^2/sum_mk;
            Y_trunc=yinter;
            Y_trunc(abs(yinter)>sqrt(alpha))=0;
            X0_temp=param.E'*Y_trunc;
            DiagMat=diag(ones(1,n3)./sqrt(mk*m));
            X0_image=X0_temp;
            X0=(reshape(X0_image,[nx*ny,n3]))*(DiagMat);
            r_big=floor(min([n/10,n3/10,m/10]));
            [Utemp, Stemp,~]=svds(X0,r_big);
            SS=diag(Stemp);
            % tic;
            E=sum(SS.^2);
            Esum=0;
            for i=1:1:r_big
                Esum=Esum+((SS(i))^2);
                if Esum >(E*0.85)
                    break
                end
            end
            r=i+1;
            r=min(r,r_big);
            U0=Utemp(:,1:r);
            Uhat=U0;
        end
        y_temp=reshape(yinter,[nx*ny,n3,nc]);
        for t = 1 : T
            Uhatm=reshape(Uhat,[nx,ny,r]);
            B = E_forw_for_AU_new(Uhatm,y_temp);
            X=reshape(Uhat*B,[nx,ny,n3]);
            Z=param.E'*((param.E*X)-yinter);
            Z_mat=reshape(Z,[nx*ny,n3]);
            Grad_U=Z_mat*B';
            if t==1
                eta=1/(7*norm(Grad_U));
            end
            Uhat_t0=Uhat;
            Uhat=Uhat-eta*Grad_U;
            [Qu,~]  =  qr(Uhat,0);
            Uhat  =  Qu(:, 1:r);
            Uhat_t1=Uhat;
            Subspace_d= ( norm((Uhat_t0 - Uhat_t1*(Uhat_t1'*Uhat_t0)), 'fro')/sqrt(r));
            if  (Subspace_d <=.01)
                break;
            end
        end
        X_GD=X+zbar_hat;
        yk=ys-(param.E*X_GD);
        global k
        % tic;
        for k=1:n3
            Ehat(:,:,k)=cgls_modi(@E_forw_for_Ak,@E_back_for_Ak,squeeze(yk(:,:,k,:)),0,1e-36,3);
        end
        
        Zhat(:,:,(ww-1)*n3+1:ww*n3)=Ehat+X_GD;
    end
    Time_Minibatch=  toc;
    NMSE_Minibatch=RMSE_modi(Zhat,Xtrue);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Subspace_Tracking_Fastest %%%%
    
    
    n3=nt/ns(ii);
    pp=n3;
    tic;
    Zhat=zeros(nx,ny,nt);
    
    param.E=getE(b1c,n3,'samp',kdata(:,:,1:n3,1)~=0);
    samp=samp1(:,:,1:n3);
    ys=kdata(:,:,1:n3,:);
    T=70;
    [zbar_hat,flag,resNE,iter] = cgls_mean(@E_forw_for_mean,@E_back_for_mean, ys,0,1e-6,3);
    ybar_hat=E_forw_for_mean(zbar_hat);
    yinter=ys-ybar_hat;
    sum_mk=nnz(ys);
    for k=1:1:n3
        mk(k)=nnz(ys(:,:,k,:));
    end
    m=max(mk);
    C_tilda=36;
    alpha=C_tilda*norm(yinter(:))^2/sum_mk;
    Y_trunc=yinter;
    Y_trunc(abs(yinter)>sqrt(alpha))=0;
    X0_temp=param.E'*Y_trunc;
    DiagMat=diag(ones(1,n3)./sqrt(mk*m));
    X0_image=X0_temp;
    X0=(reshape(X0_image,[nx*ny,n3]))*(DiagMat);
    r_big=floor(min([n/10,n3/10,m/10]));
    [Utemp, Stemp,~]=svds(X0,r_big);
    SS=diag(Stemp);
    % tic;
    E=sum(SS.^2);
    Esum=0;
    for i=1:1:r_big
        Esum=Esum+((SS(i))^2);
        if Esum >(E*0.85)
            break
        end
    end
    r=i+1;
    r=min(r,r_big);
    U0=Utemp(:,1:r);
    Uhat=U0;
    y_temp=reshape(yinter,[nx*ny,n3,nc]);
    for t = 1 : T
        Uhatm=reshape(Uhat,[nx,ny,r]);
        B = E_forw_for_AU_new(Uhatm,y_temp);
        X=reshape(Uhat*B,[nx,ny,n3]);
        Z=param.E'*((param.E*X)-yinter);
        Z_mat=reshape(Z,[nx*ny,n3]);
        Grad_U=Z_mat*B';
        if t==1
            eta=1/(7*norm(Grad_U));
        end
        Uhat_t0=Uhat;
        Uhat=Uhat-eta*Grad_U;
        [Qu,~]  =  qr(Uhat,0);
        Uhat  =  Qu(:, 1:r);
        Uhat_t1=Uhat;
        Subspace_d= ( norm((Uhat_t0 - Uhat_t1*(Uhat_t1'*Uhat_t0)), 'fro')/sqrt(r));
        if  (Subspace_d <=.01)
            break;
        end
    end
    X_GD=X+zbar_hat;
    yk=ys-(param.E*X_GD);
    global k
    % tic;
    for k=1:n3
        Ehat(:,:,k)=cgls_modi(@E_forw_for_Ak,@E_back_for_Ak,squeeze(yk(:,:,k,:)),0,1e-36,3);
    end
    
    Zhat(:,:,1:n3)=Ehat+X_GD;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param.E=getE(b1c,nt-n3,'samp',kdata(:,:,n3+1:nt,1)~=0);
    samp=samp1(:,:,n3+1:nt);
    ys=kdata(:,:,n3+1:nt,:);
    ybar_hat=param.E*repmat(zbar_hat,[1,1,nt-n3]);
    yinter=ys-ybar_hat;
    y_temp=reshape(yinter,[nx*ny,nt-n3,nc]);
    Uhatm=reshape(Uhat,[nx,ny,r]);
    AUtmp2 = E_forw_for_AU_online(Uhatm,yinter);
    for k=1:1:nt-n3
        mk(k)=nnz(ys(:,:,k,:));
    end
    m=max(mk);
    global kk
    for kk=1:1:nt-n3
        ObsFreq=find(samp(:,:,kk));
        m1 = mk(kk);
        AUk_tmp = AUtmp2(ObsFreq,:,: ) ;
        yk_tmp=y_temp(ObsFreq,kk,: ) ;
        AUk = reshape(AUk_tmp, [m1, r]);
        yk = reshape(yk_tmp, [m1,1]);
        B=AUk\yk;
        Xhat=Uhat*B;
        param.E=getE(b1c,1,'samp',kdata(:,:,kk+n3,1)~=0);
        Yhathat=ys(:,:,kk,:)-(param.E*reshape(Xhat,[nx,ny,1]));
        Ehat=cgls_modi(@E_forw_for_Ak_online,@E_back_for_Ak_online,squeeze(Yhathat),0,1e-36,3);
        Zhat(:,:,n3+kk)=zbar_hat+reshape(Xhat,[nx,ny,1])+Ehat;
    end
    Time_Online=  toc;
    NMSE_Online=RMSE_modi(Zhat,Xtrue);
    
    fprintf(fid, '%d & %8.4f (%5.2f) & %8.4f (%5.2f)\n', ns(ii),NMSE_Minibatch,Time_Minibatch,NMSE_Online,Time_Online);
end
fclose(fid);

function B = E_forw_for_AU_new(U_im,y_temp)
global nx ny n3 nc r b1c samp mk
B=zeros(r,n3);
n=nx*ny;
smaps = b1c;
mask1=samp;
% U_im = reshape(Uhat,[nx,ny,r]);
U_im2 = reshape(U_im,[ nx, ny, 1,r]);  % nx ny 1 r
smaps = reshape(smaps, [nx,ny,nc,1]); % nx ny nc 1
AUtmp = bsxfun(@times, U_im2, smaps);  % nx ny nc r
AUtmp1= fft2c_mri(AUtmp); %nx ny nc r
AUtmp2=reshape(AUtmp1,[nx*ny,nc,r]);
for k=1:1:n3
    ObsFreq=find(mask1(:,:,k));
    m1 = mk(k);
    AUk_tmp = AUtmp2(ObsFreq,:,: ) ;
    yk_tmp=y_temp(ObsFreq,k,: ) ;
    AUk = reshape(AUk_tmp, [m1, r]);
    yk = reshape(yk_tmp, [m1,1]);
    B(:,k)=AUk\yk;
end
end


function AUtmp2 = E_forw_for_AU_online(U_im,y_temp)
global nx ny  nc r b1c samp mk
n=nx*ny;
smaps = b1c;
mask1=samp;
% U_im = reshape(Uhat,[nx,ny,r]);
U_im2 = reshape(U_im,[ nx, ny, 1,r]);  % nx ny 1 r
smaps = reshape(smaps, [nx,ny,nc,1]); % nx ny nc 1
AUtmp = bsxfun(@times, U_im2, smaps);  % nx ny nc r
AUtmp1= fft2c_mri(AUtmp); %nx ny nc r
AUtmp2=reshape(AUtmp1,[nx*ny,nc,r]);
end

function Amat_zbar  = E_forw_for_mean(zbar) %zbar: nx ny
global nx ny n3 nc  b1c samp

smaps = b1c; % nx ny nc
samp = reshape(samp, [nx,ny,n3,1]) ; %nx ny n3 1
s = bsxfun(@times, zbar, smaps);  % nx ny nc
S=fft2c_mri(s); %nx ny nc
Spermute = reshape(S, [nx,ny,1,nc]);  % nx ny 1 nc
Amat_zbar = bsxfun(@times,Spermute,samp);  % nx ny n3 nc
end


function zbar_hat  = E_back_for_mean(Y_in) %Y_in: nx ny nt nc
global nx ny n3 nc  b1c samp
smaps = b1c; % nx ny nc
samp = reshape(samp, [nx,ny,n3,1]) ; %nx ny nt 1

S = bsxfun(@times,Y_in,samp); %nx ny n3 nc
s = ifft2c_mri(S);  %nx ny n3 nc
zbar_hat = sum(bsxfun(@times,s,reshape(conj(smaps),[nx,ny,1,nc])),4);  %nx ny n3 after the sum step
end
function y=SoftThresh(x,p)
y=(abs(x)-p).*sign(x).*(abs(x)>p);
y(isnan(y))=0;
end

function Ak = E_forw_for_Ak(xk)
global nx ny nt nc r b1c samp k
s = zeros(nx,ny,nc);

% smaps_k = squeeze(arg.smaps(:,:,k,:)); %nx ny nc
samp_k = squeeze(samp(:,:,k)) ;  %nx ny
s = bsxfun(@times,xk,b1c); % nx ny  nc

S=fft2c_mri(s); % nx ny 1 nc
Ak = bsxfun(@times,S,samp_k);
end


function x = E_back_for_Ak(zk)
global nx ny nt nc r b1c samp k
zk_2 = squeeze(zk); %zk: nx ny nc
x = zeros(nx,ny);
% smaps_k = squeeze(arg.smaps(:,:,k,:)); %nx ny nc
samp_k = samp(:,:,k) ;  %nx ny

ztmp = bsxfun(@times,zk_2,samp_k); % nx ny nc
s = ifft2c_mri(ztmp); %nx ny nc
x = sum(bsxfun(@times,s,conj(b1c)),3);
end


function Ak = E_forw_for_Ak_online(xk)
global nx ny nt nc r b1c samp kk
s = zeros(nx,ny,nc);

% smaps_k = squeeze(arg.smaps(:,:,k,:)); %nx ny nc
samp_k = squeeze(samp(:,:,kk)) ;  %nx ny
s = bsxfun(@times,xk,b1c); % nx ny  nc

S=fft2c_mri(s); % nx ny 1 nc
Ak = bsxfun(@times,S,samp_k);
end


function x = E_back_for_Ak_online(zk)
global nx ny nt nc r b1c samp k kk
zk_2 = squeeze(zk); %zk: nx ny nc
x = zeros(nx,ny);
% smaps_k = squeeze(arg.smaps(:,:,k,:)); %nx ny nc
samp_k = samp(:,:,kk) ;  %nx ny

ztmp = bsxfun(@times,zk_2,samp_k); % nx ny nc
s = ifft2c_mri(ztmp); %nx ny nc
x = sum(bsxfun(@times,s,conj(b1c)),3);
end