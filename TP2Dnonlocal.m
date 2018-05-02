%%% TARGET PATTERNS IN 2D ARRAY OF OSCILLATORS/ NONLOCAL COUPLING
%%% Author: Gabriela Jaramillo
%%%================================================================
% u_t = L*u - Om*|J*u|^2 + eps*g(x,y)
% Here u(x,y)= phase of oscillator at position (x,y)
% L and J are convolution kernels
% The function g(x,y) is an inhomogeneity at the center of the domain
% This code is based on Aly-Khan Kassam's ETD code from his paper 
% " Solving Reaction-Diffusion Equations 10 Times Faster"
%========================= CUSTOM SET UP ==========================

 lam=100; N=2^9; D=1; h=0.5; Om=1; eps=-0.5; %domain, grid size, diff par, timestep, transp par, epsilon
%========================== Cartesian  ============================
%%% Uncoment this section for g(x,y) with elliptical core.

%xx=(lam/N)*(1:N)'; [XX,YY]=ndgrid(xx,xx); 
%g3 = 10*eps./( 1 + (1/30)*(XX -lam/2).^2 + ( YY -lam/2).^2 ).^(3/2); 
% hatg3= fftn(g3); %inhomogeneity

%========================== Polar  ================================
%%% Uncoment this section for g(x,y) with "flower" cores.
 xx=(lam/N )*(-N/2:N/2-1)'; [XX,YY]=ndgrid(xx,xx); 
 [T,R] = cart2pol(XX,YY);
 g3 = 10*eps*(cos(4*T)+1)./((1+R).^(3)) ; hatg3 = fftn(g3); %inhomogeneity
%===================== Initial Parameters ===========================
u=rand(N,N)*0.1; v=fftn(u); %random IC . 
k=[0:N/2 -N/2+1:-1]'/(lam/(2*pi)); %wave numbers
[xi,eta]=ndgrid(k,k); %2D wave numbers. 
L=sparse(-D*(eta.^2+xi.^2)./(1 + eta.^2 +xi.^2) ); %Diffusive Kernel
Fr= false(N,1); %High frequencies for de-aliasing
Fr((N/2+1-round(N/6) :1+ N/2+round(N/6)))=1;
[alxi,aleta]=ndgrid(Fr,Fr); 
ind=alxi | aleta; 
%=============== PRECOMPUTING ETDRK4 COEFFS =====================
E=exp(h*L); E2=exp(h*L/2);
M=16; % no. of points for complex mean
r=exp(1i*pi*((1:M)-0.5)/M); % roots of unity
L=L(:); LR=h*L(:,ones(M,1))+r(ones(N^2,1),:); 
Q=h*real(mean( (exp(LR/2)-1)./LR ,2));
f1=h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2=h*real(mean( (4+2*LR+exp(LR).*(-4+2*LR))./LR.^3 ,2));
f3=h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
f1=reshape(f1,N,N); f2=reshape(f2,N,N); f3=reshape(f3,N,N);
Q=reshape(Q,N,N); clear LR

%================== TIME STEPPING LOOP (NON Radial case)==================
tmax=80; nmax=round(tmax/h); %Max time, # iterations.
uunonr = u;
weta = 1i*eta.*v./(1+eta.^2+xi.^2); %nonlinearity
wxi = 1i*xi.*v./(1+eta.^2+xi.^2);

for n = 1:nmax
%t=n*h; %**Nonlinear terms are evaluated in physical space**
Nv=Om*fftn( -real(ifftn(weta) ).^2  - real( ifftn(wxi) ).^2 ) + hatg3; 
a=E2.*v + Q.*Nv; %Coefficient ?a? in ETDRK formula
Na= Om*fftn(  -real(ifftn(1i*eta.*a./(1+eta.^2+xi.^2))).^2 -real(ifftn(1i*xi.*a./(1+eta.^2+xi.^2))).^2 )+ hatg3; 
b=E2.*v + Q.*Na; %Coefficient ?b? in ETDRK formula
Nb=Om*fftn( -real(ifftn(1i*eta.*b./(1+eta.^2+xi.^2))).^2 -real(ifftn(1i*xi.*b./(1+eta.^2+xi.^2))).^2 )+ hatg3 ; 
c=E2.*a + Q.*(2*Nb-Nv); %Coefficient ?c? in ETDRK formula
Nc=Om*fftn(  -real(ifftn(1i*eta.*c./(1+eta.^2+xi.^2))).^2 -real(ifftn(1i*xi.*c./(1+eta.^2+xi.^2))).^2 )+ hatg3; 
v=E.*v + Nv.*f1 + (Na+Nb).*f2 + Nc.*f3; %update
weta = 1i*eta.*v./(1+eta.^2+xi.^2);
wxi = 1i*xi.*v./(1+eta.^2+xi.^2);
v(ind) = 0; % High frequency removal --- de-aliasing
weta(ind) =0;
wxi(ind) = 0;
uunonr(:,:,n+1) = real(ifftn(v));
end

%========================= Movie / Plots ===============================
%%% This movie and plots show cos(u(x,y))
close all; clear('M');
M = VideoWriter('Sin3TV2.mov');
open(M);

for ii = 1:tmax
    
surf(XX,YY, cos(uunonr(:,:,ii/h))  ), shading interp, lighting phong, axis equal
view([-90 90]); set(gca,'zlim',[-100 1])
%light('color',[1 1 0],'position',[-1,2,2])
material([0.30 0.60 0.60 40.00 1.00]);

frame = getframe;
writeVideo(M,frame);
end
close(M)


figure(1)
clf
surf(XX,YY, cos(uunonr(:,:,end))  ), shading interp, lighting phong, axis equal
view([-90 90]); set(gca,'zlim',[-100 1])
%light('color',[1 1 0],'position',[-1,2,2])
material([0.30 0.60 0.60 40.00 1.00])



