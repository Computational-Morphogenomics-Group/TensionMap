function [qF, ERes] = main_minimization(bCells, d0, rBX, rBY, q0, avgB)
    
    q = q0(:,1:2);
    p = q0(:,4);

    Aeq = [[zeros(1,3*size(q,1)),ones(1,size(q,1))/size(q,1)]; ...
           [zeros(1,2*size(q,1)),ones(1,size(q,1))/size(q,1),zeros(1,size(q,1))]]; 
       
    beq = [mean(p);mean(q0(:,3))];

    optimset = optimoptions('fmincon','Display','none','MaxFunEvals',1e6,'MaxIter',2e3,'TolFun',1e-6,...
               'Algorithm','interior-point','GradObj','on','GradConstr','on','DerivativeCheck','off',...
               'Hessian','on','HessFcn',@(x,lambda) hessian(x,lambda,bCells,d0,rBX,rBY,avgB));
    lb = -inf*ones(size(q0));
    lb(:,4) = .001;
    ub = inf*ones(size(q0));
    ub(:,4) = 1000;

    [qF,ERes] = fmincon(@(q) energy(q, d0, bCells, rBX, rBY), ...
                q0,[],[],Aeq,beq,lb,ub,@(q) ensurePositive(q,bCells,d0),optimset);
end

function [ H ] = hessian(x,lambda,bCells,d0,RBx,RBy,avgB)

    Q = reshape(x,length(x)/4,4);
    NB = size(d0,1);
    NC = size(d0,2);
    
    q = Q(:,1:2);
    theta = Q(:,3);
    p = Q(:,4);
    
    dP = p(bCells(:,1)) - p(bCells(:,2));
    dT = theta(bCells(:,1)) - theta(bCells(:,2));
    dQ = q(bCells(:,1),:) - q(bCells(:,2),:);
    QL = sum(dQ.^2,2);
    
    rho = bsxfun(@rdivide,d0*bsxfun(@times,p,q),dP);
    Rsq = (p(bCells(:,1)).*p(bCells(:,2)).*QL - (dP .* dT))  ./ dP.^2;
    indZ = Rsq <= 0;
    R = sqrt( Rsq );
    R(indZ) = 0;
    indZ = ~indZ;
    
    deltaX = -bsxfun(@minus,RBx,rho(:,1));
    deltaY = -bsxfun(@minus,RBy,rho(:,2));
    dMag = sqrt(deltaX.^2 + deltaY.^2);
    deltaX = deltaX ./ dMag;
    deltaY = deltaY ./ dMag;

    d = bsxfun(@minus,dMag,R);

    % Compute gradients.
    [ dRhoX, dRhoY, dR ] = returnBondGrads( q, theta, p, bCells );
    dR(~indZ,:) = 0;
    
    %% Build objective hessian.
    % 1a + 2c
    rRatio = bsxfun(@rdivide,R,dMag);
    dNormXX = sum( rRatio .*( (deltaX .* deltaX) ), 2);
    dNormXY = sum( rRatio .*( (deltaX .* deltaY) ), 2);
    dNormYY = sum( rRatio .*( (deltaY .* deltaY) ), 2);

    dRhoX = reshape(dRhoX',[1,8,NB]);
    dRhoY = reshape(dRhoY',[1,8,NB]);
    dR = reshape(dR',[1,8,NB]);

    dRhoXT = permute(dRhoX,[2,1,3]);
    dRhoYT = permute(dRhoY,[2,1,3]);
    dRT = permute(dR,[2,1,3]);
    
    Hobj = bsxfun(@times,dRhoXT,permute(bsxfun(@times,dNormXX,permute(dRhoX,[3,1,2])),[2,3,1])) + ...
           bsxfun(@times,dRhoXT,permute(bsxfun(@times,dNormXY,permute(dRhoY,[3,1,2])),[2,3,1])) + ...
           bsxfun(@times,dRhoYT,permute(bsxfun(@times,dNormXY,permute(dRhoX,[3,1,2])),[2,3,1])) + ...
           bsxfun(@times,dRhoYT,permute(bsxfun(@times,dNormYY,permute(dRhoY,[3,1,2])),[2,3,1]));

    % 1b    
    dNormX = indZ .* sum( deltaX , 2);
    dNormY = indZ .* sum( deltaY , 2);
     
    Hobj = Hobj - bsxfun(@times,dRT,permute(bsxfun(@times,dNormX,permute(dRhoX,[3,1,2])),[2,3,1])) - ...
           bsxfun(@times,dRT,permute(bsxfun(@times,dNormY,permute(dRhoY,[3,1,2])),[2,3,1])) - ... 
           bsxfun(@times,dRhoXT,permute(bsxfun(@times,dNormX,permute(dR,[3,1,2])),[2,3,1])) - ...
           bsxfun(@times,dRhoYT,permute(bsxfun(@times,dNormY,permute(dR,[3,1,2])),[2,3,1]));
       
    % 1c
    nPix = indZ .* (avgB.*ones(size(bCells,1),1));
    Hobj = Hobj + bsxfun(@times,dRT,permute(bsxfun(@times,nPix,permute(dR,[3,1,2])),[2,3,1]));
       
    % 2b
    avgRatio = sum( (d./dMag), 2);
    Hobj = Hobj + bsxfun(@times,dRhoXT,permute(bsxfun(@times,avgRatio,permute(dRhoX,[3,1,2])),[2,3,1])) + ...
                  bsxfun(@times,dRhoYT,permute(bsxfun(@times,avgRatio,permute(dRhoY,[3,1,2])),[2,3,1]));
    Hobj = permute(Hobj,[3,1,2])/NB;
    
    % Return bond hessians. 2a + 2d + cons
    dNormX = sum( deltaX .* d, 2 ) / NB;
    dNormY = sum( deltaY .* d, 2 ) / NB;
    dAvg = indZ .* sum( d, 2 ) / NB;
    
    [ bondHess ] = returnBondHessian( q, theta, p, bCells, dNormX, dNormY, dAvg, lambda.ineqnonlin ); 
    H = Hobj + bondHess;
    H = loadHessian(H,bCells,NC,8);
    H = H + H';
    
end

function [ c, ceq, dc, dceq ] = ensurePositive(Q,bCells,d0) 
    
    q = Q(:,1:2);
    theta = Q(:,3);
    p = Q(:,4);

    ceq = [];
    c = ((d0*p) .* (d0*theta)) - p(bCells(:,1)).*p(bCells(:,2)).*sum((d0*q).^2,2);
    
    if (nargout > 2)
        dP = d0*p;
        dT = d0*theta;
        dQ = d0*q;
        QL = sum(dQ.^2,2);
        
        % X gradient
        gX = -2*(p(bCells(:,1)).*p(bCells(:,2)).*dQ(:,1));
        gX = bsxfun(@times,gX,d0);
        
        % Y gradient
        gY = -2*(p(bCells(:,1)).*p(bCells(:,2)).*dQ(:,2));
        gY = bsxfun(@times,gY,d0);

        % T gradient
        gTh = bsxfun(@times,dP,d0);
        
        % P gradient
        gP = bsxfun(@times,dT,d0);
        gP = gP - bsxfun(@rdivide,bsxfun(@times,(QL.*p(bCells(:,1)).*p(bCells(:,2))),abs(d0))',p)';
        
        dc = [gX,gY,gTh,gP]';
        dceq = [];
    end
end

function [ E, dE ] = energy( Q, d0, bCells, RBx, RBy)
    % FORCE BALANCE ENERGY 
    
    q = Q(:,1:2);
    theta = Q(:,3);
    p = Q(:,4);
    
    dP = p(bCells(:,1)) - p(bCells(:,2));
    dT = theta(bCells(:,1)) - theta(bCells(:,2));
    dQ = q(bCells(:,1),:) - q(bCells(:,2),:);
    QL = sum(dQ.^2,2);
    
    rho = bsxfun(@rdivide,d0*bsxfun(@times,p,q),dP);
    Rsq = ( p(bCells(:,1)).*p(bCells(:,2)).*QL - (dP .* dT) ) ./ dP.^2;
    R = sqrt(Rsq);
    indZ = Rsq<=0;
    R(indZ) = 0;
    
    deltaX = -bsxfun(@minus,RBx,rho(:,1));
    deltaY = -bsxfun(@minus,RBy,rho(:,2));
    dMag = sqrt(deltaX.^2 + deltaY.^2);
    d = bsxfun(@minus,dMag,R);

    E = .5*mean( sum((d.^2),2) ); 
    
    if (nargout == 2)
        NB = size(d0,1);
        NC = size(d0,2);
        [ dRhoX, dRhoY, dR ] = returnBondGrads( q, theta, p, bCells );
        dNormX = sum( (deltaX .* (d ./ dMag)),2);
        dNormY = sum( (deltaY .* (d ./ dMag)),2);
        avgD = sum(d,2);
        indZ = ~indZ;
        dR(~indZ,:) = 0;
        dE = (bsxfun(@times,dNormX,dRhoX) + bsxfun(@times,dNormY,dRhoY) - bsxfun(@times,avgD,dR)) / NB;
        rows = [bCells(:,1);bCells(:,1)+NC;bCells(:,1)+2*NC;bCells(:,1)+3*NC;bCells(:,2);bCells(:,2)+NC;bCells(:,2)+2*NC;bCells(:,2)+3*NC];
        vals = dE(:)';
        dE = accumarray(rows,vals,[4*NC,1]);
    end
end

function [ dRhoX, dRhoY, dR ] = returnBondGrads( q, theta, p, bCells )
    % RETURN BOND GRADS 

    p1 = p(bCells(:,1));
    p2 = p(bCells(:,2));
    q1x = q(bCells(:,1),1);
    q2x = q(bCells(:,2),1);
    q1y = q(bCells(:,1),2);
    q2y = q(bCells(:,2),2);
    t1 = theta(bCells(:,1));
    t2 = theta(bCells(:,2));
    
    dRhoX = rhoXGrad(p1,p2,q1x,q2x);
    dRhoY = rhoYGrad(p1,p2,q1y,q2y);
    dR = radiusGrad(p1,p2,q1x,q2x,q1y,q2y,t1,t2);
end

function [ h ] = returnBondHessian( q, theta, p, bCells, dNormX, dNormY, dAvg, lambda )
    
    %% Calculculate elements of Rho and Radius Hessian
    hRhoX = rhoXHess(p(bCells(:,1)),p(bCells(:,2)),q(bCells(:,1),1),q(bCells(:,2),1));
    hRhoY = rhoYHess(p(bCells(:,1)),p(bCells(:,2)),q(bCells(:,1),2),q(bCells(:,2),2));
    hR = radiusHess(p(bCells(:,1)),p(bCells(:,2)),q(bCells(:,1),1),...
                 q(bCells(:,2),1),q(bCells(:,1),2),q(bCells(:,2),2),theta(bCells(:,1)),theta(bCells(:,2)));
    hCon = constHess( p(bCells(:,1)), p(bCells(:,2)), q(bCells(:,1),1),...
                                q(bCells(:,2),1), q(bCells(:,1),2), q(bCells(:,2),2));

    hRhoX = reshape(hRhoX,[size(hRhoX,1),8,8]);
    hRhoY = reshape(hRhoY,[size(hRhoY,1),8,8]);
    hR = reshape(hR,[size(hR,1),8,8]);
    hCon = reshape(hCon,[size(hCon,1),8,8]);

    h = bsxfun(@times,hRhoX,dNormX) + bsxfun(@times,hRhoY,dNormY) - ...
        bsxfun(@times,hR,dAvg) + bsxfun(@times,hCon,lambda);  
end

function [ h ] = loadHessian( h, bCells, NC, Z )
%LOADHESSIAN Summary of this function goes here
%   Detailed explanation goes here

    pF = Z*(Z+1)/2;
    NB = size(bCells,1);
    row = zeros(pF*NB,1);
    col = zeros(pF*NB,1);
    val = zeros(pF*NB,1);
    n = 1;
    delta = Z/2 + 1;
    for ii = 1:Z
        for jj = ii:Z
            
            II = ((n-1)*NB) + (1:NB);
            row(II) = 1:NB;
            
            if (ii <= Z/2)
                row(II) = bCells(:,1) + (ii-1)*NC;
            else
                row(II) = bCells(:,2) + (ii-delta)*NC;
            end
            
            if (jj <= Z/2)
                col(II) = bCells(:,1) + (jj-1)*NC;
            else
                col(II) = bCells(:,2) + (jj-delta)*NC;
            end

            val(II) = h(:,ii,jj);
            
            n = n + 1;
        end
    end
    
    h = sparse(row(val~=0),col(val~=0),val(val~=0),Z/2*NC,Z/2*NC);
end

function Dx = rhoXGrad(p1,p2,q1x,q2x)
    %RHOXGRAD
    %    DX = RHOXGRAD(P1,P2,Q1X,Q2X)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    04-Jul-2017 12:00:25
    
    t2 = p1-p2;
    t3 = 1.0./t2;
    t4 = 1.0./t2.^2;
    t5 = p1.*q1x;
    t6 = t5-p2.*q2x;
    z = zeros(size(p1));
    Dx = [p1.*t3,z,z,q1x.*t3-t4.*t6,-p2.*t3,z,z,-q2x.*t3+t4.*t6];
end

function Dy = rhoYGrad(p1,p2,q1y,q2y)
    %RHOYGRAD
    %    DY = RHOYGRAD(P1,P2,Q1Y,Q2Y)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    04-Jul-2017 12:00:26
    
    t2 = p1-p2;
    t3 = 1.0./t2;
    t4 = 1.0./t2.^2;
    t5 = p1.*q1y;
    t6 = t5-p2.*q2y;
    z = zeros(size(p1));
    Dy = [z,p1.*t3,z,q1y.*t3-t4.*t6,z,-p2.*t3,z,-q2y.*t3+t4.*t6];
end

function dR = radiusGrad(p1,p2,q1x,q2x,q1y,q2y,t1,t2)
    %RADIUSGRAD
    %    H = RADIUSGRAD(P1,P2,Q1X,Q2X,Q1Y,Q2Y,T1,T2)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    04-Jul-2017 12:00:52
    
    t4 = p1-p2;
    t5 = q1x-q2x;
    t6 = q1y-q2y;
    t7 = 1.0./t4.^2;
    t8 = t1-t2;
    t9 = t4.*t8;
    t10 = t5.^2;
    t11 = t6.^2;
    t12 = t10+t11;
    t15 = p1.*p2.*t12;
    t13 = t9-t15;
    t16 = t7.*t13;
    t14 = 1.0./sqrt(-t16);
    t17 = q1x.*2.0;
    t18 = q2x.*2.0;
    t19 = t17-t18;
    t20 = p1.*p2.*t7.*t14.*t19.*(1.0./2.0);
    t21 = q1y.*2.0;
    t22 = q2y.*2.0;
    t23 = t21-t22;
    t24 = p1.*p2.*t7.*t14.*t23.*(1.0./2.0);
    t25 = 1.0./t4;
    t26 = 1.0./t4.^3;
    t27 = t13.*t26.*2.0;
    
    dR = [t20,t24,t14.*t25.*(-1.0./2.0),...
         t14.*(t27+t7.*(-t1+t2+p2.*t12)).*(1.0./2.0),...
        -t20,-t24,t14.*t25.*(1.0./2.0),...
         t14.*(t27-t7.*(t1-t2+p1.*t12)).*(-1.0./2.0)];
end

function Hx = rhoXHess(p1,p2,q1x,q2x)
    %RHOXHESS
    %    HX = RHOXHESS(P1,P2,Q1X,Q2X)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    04-Jul-2017 11:22:44
    
    t2 = p1-p2;
    t3 = 1.0./t2.^2;
    t4 = p1.*t3;
    t5 = 1.0./t2;
    t6 = 1.0./t2.^3;
    t7 = p1.*q1x;
    t11 = p2.*q2x;
    t8 = t7-t11;
    t9 = t6.*t8.*2.0;
    t10 = p2.*t3;
    t12 = q1x.*t3;
    t13 = q2x.*t3;
    t14 = -t9+t12+t13;
    t15 = -t5-t10;
    z = zeros(size(p1));
    Hx = [z,z,z,-t4+t5,z,z,z,t4,...
          z,z,z,z,z,z,z,z,...
          z,z,z,z,z,z,z,z,...
          t5-p1.*t3,z,z,t9-q1x.*t3.*2.0,t10,z,z,t14,...
          z,z,z,t10,z,z,z,t15,...
          z,z,z,z,z,z,z,z,...
          z,z,z,z,z,z,z,z,...
          t4,z,z,t14,t15,z,z,t9-q2x.*t3.*2.0];
end

function Hy = rhoYHess(p1,p2,q1y,q2y)
    %RHOYHESS
    %    HY = RHOYHESS(P1,P2,Q1Y,Q2Y)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    04-Jul-2017 11:22:45
    
    t2 = p1-p2;
    t3 = 1.0./t2.^2;
    t4 = p1.*t3;
    t5 = 1.0./t2;
    t6 = 1.0./t2.^3;
    t7 = p1.*q1y;
    t11 = p2.*q2y;
    t8 = t7-t11;
    t9 = t6.*t8.*2.0;
    t10 = p2.*t3;
    t12 = q1y.*t3;
    t13 = q2y.*t3;
    t14 = -t9+t12+t13;
    t15 = -t5-t10;
    z = zeros(size(p1));
    Hy = [z,z,z,z,z,z,z,z,...
          z,z,z,-t4+t5,z,z,z,t4,...
          z,z,z,z,z,z,z,z,...
          z,t5-p1.*t3,z,t9-q1y.*t3.*2.0,z,t10,z,t14,...
          z,z,z,z,z,z,z,z,...
          z,z,z,t10,z,z,z,t15,...
          z,z,z,z,z,z,z,z,...
          z,t4,z,t14,z,t15,z,t9-q2y.*t3.*2.0];
end

function H = radiusHess(p1,p2,q1x,q2x,q1y,q2y,t1,t2)
    %RADIUSHESS
    %    H = RADIUSHESS(P1,P2,Q1X,Q2X,Q1Y,Q2Y,T1,T2)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    03-Jul-2017 18:13:17
    
    t4 = p1-p2;
    t5 = q1x-q2x;
    t6 = q1y-q2y;
    t7 = 1.0./t4.^2;
    t17 = q1x.*2.0;
    t18 = q2x.*2.0;
    t8 = t17-t18;
    t9 = t1-t2;
    t10 = t4.*t9;
    t11 = t5.^2;
    t12 = t6.^2;
    t13 = t11+t12;
    t19 = p1.*p2.*t13;
    t14 = t10-t19;
    t15 = p1.^2;
    t16 = p2.^2;
    t22 = t7.*t14;
    t20 = 1.0./(-t22).^(3.0./2.0);
    t21 = 1.0./t4.^4;
    t23 = 1.0./sqrt(-t22);
    t24 = 1.0./t4.^3;
    t25 = t8.^2;
    t26 = q1y.*2.0;
    t27 = q2y.*2.0;
    t28 = t26-t27;
    t29 = p1.*p2.*t8.*t20.*t24.*(1.0./4.0);
    t30 = t14.*t24.*2.0;
    t31 = t8.*t15.*t16.*t20.*t21.*t28.*(1.0./4.0);
    t32 = p2.*t13;
    t33 = -t1+t2+t32;
    t34 = t7.*t33;
    t35 = t30+t34;
    t36 = p1.*p2.*t7.*t23;
    t37 = t28.^2;
    t38 = p1.*p2.*t20.*t24.*t28.*(1.0./4.0);
    t39 = p1.*t13;
    t40 = t1-t2+t39;
    t43 = t7.*t40;
    t41 = t30-t43;
    t42 = t7.*t23.*(1.0./2.0);
    t44 = 1.0./t4;
    t45 = t20.*t35.*t44.*(1.0./4.0);
    t46 = t42+t45;
    t47 = p2.*t7.*t8;
    t69 = p1.*p2.*t8.*t24.*2.0;
    t48 = t47-t69;
    t49 = t23.*t48.*(1.0./2.0);
    t50 = p2.*t7.*t28;
    t70 = p1.*p2.*t24.*t28.*2.0;
    t51 = t50-t70;
    t52 = t23.*t51.*(1.0./2.0);
    t53 = t14.*t21.*6.0;
    t54 = t15.*t16.*t20.*t21.*t25.*(1.0./4.0);
    t55 = p2.*t7.*t8.*t23.*(1.0./2.0);
    t56 = p1.*p2.*t8.*t23.*t24;
    t57 = p1.*p2.*t7.*t8.*t20.*t35.*(1.0./4.0);
    t58 = p1.*t7.*t8.*t23.*(1.0./2.0);
    t59 = p1.*p2.*t7.*t8.*t20.*t41.*(1.0./4.0);
    t60 = t15.*t16.*t20.*t21.*t37.*(1.0./4.0);
    t61 = -t36+t60;
    t62 = p2.*t7.*t23.*t28.*(1.0./2.0);
    t63 = p1.*p2.*t23.*t24.*t28;
    t64 = p1.*p2.*t7.*t20.*t28.*t35.*(1.0./4.0);
    t65 = p1.*t7.*t23.*t28.*(1.0./2.0);
    t66 = p1.*p2.*t7.*t20.*t28.*t41.*(1.0./4.0);
    t67 = t7.*t20.*(1.0./4.0);
    t68 = -t42-t45;
    t71 = t20.*t41.*t44.*(1.0./4.0);
    t72 = t24.*t33.*2.0;
    t73 = t7.*t13;
    t74 = t53+t72+t73-t24.*t40.*2.0;
    t75 = t23.*t74.*(1.0./2.0);
    t76 = t20.*t35.*t41.*(1.0./4.0);
    t77 = t75+t76;
    t78 = p1.*t7.*t8;
    t79 = t69+t78;
    t80 = t23.*t79.*(1.0./2.0);
    t81 = p1.*t7.*t28;
    t82 = t70+t81;
    t83 = t23.*t82.*(1.0./2.0);
    t84 = t42+t71;
    H = [p1.*p2.*t7.*1.0./sqrt(-t7.*t14)-t15.*t16.*t20.*t21.*t25.*(1.0./4.0),-t31,t29,t49-p1.*p2.*t7.*t8.*t20.*t35.*(1.0./4.0),-t36+t54,t31,-t29,t59+t80,t8.*t15.*t16.*t20.*t21.*t28.*(-1.0./4.0),t36-t15.*t16.*t20.*t21.*t37.*(1.0./4.0),t38,t52-p1.*p2.*t7.*t20.*t28.*t35.*(1.0./4.0),t31,t61,-t38,t66+t83,t29,t38,t7.*t20.*(-1.0./4.0),t46,-t29,-t38,t67,-t42-t71,t55-p1.*p2.*t8.*t23.*t24-p1.*p2.*t7.*t8.*t20.*t35.*(1.0./4.0),t62-p1.*p2.*t23.*t24.*t28-p1.*p2.*t7.*t20.*t28.*t35.*(1.0./4.0),t46,t20.*t35.^2.*(-1.0./4.0)-t23.*(t53+t24.*t33.*4.0).*(1.0./2.0),-t55+t56+t57,-t62+t63+t64,t68,t77,t54-p1.*p2.*t7.*t23,t31,-t29,-t49+t57,t36-t54,-t31,t29,-t59-t80,t31,t61,-t38,-t52+t64,-t31,t36-t60,t38,-t66-t83,-t29,-t38,t67,t68,t29,t38,-t67,t84,t56+t58+t59,t63+t65+t66,-t42-t20.*t41.*t44.*(1.0./4.0),t77,-t56-t58-t59,-t63-t65-t66,t84,t20.*t41.^2.*(-1.0./4.0)-t23.*(t53-t24.*t40.*4.0).*(1.0./2.0)];
end

function H = constHess(p1,p2,q1x,q2x,q1y,q2y)
    %GENERATECONSTHESSIAN
    %    H = GENERATECONSTHESSIAN(P1,P2,Q1X,Q2X,Q1Y,Q2Y)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    03-Jul-2017 20:34:58
    
    t2 = q1x.*2.0;
    t3 = q2x.*2.0;
    t4 = t2-t3;
    t5 = p1.*p2.*2.0;
    t6 = q1y.*2.0;
    t7 = q2y.*2.0;
    t8 = t6-t7;
    t9 = q1x-q2x;
    t10 = q1y-q2y;
    t11 = p2.*t4;
    t12 = p2.*t8;
    t13 = p1.*t4;
    t14 = p1.*t8;
    t15 = t9.^2;
    t16 = t10.^2;
    t17 = -t15-t16;
    
    H =     [p1.*p2.*-2.0,zeros(size(p1)),zeros(size(p1)),-p2.*t4,t5,zeros(size(p1)),zeros(size(p1)),...
        -t13,zeros(size(p1)),-t5,zeros(size(p1)),-p2.*t8,zeros(size(p1)),t5,zeros(size(p1)),-t14,...
        zeros(size(p1)),zeros(size(p1)),zeros(size(p1)),ones(size(p1)),zeros(size(p1)),zeros(size(p1)),...
        zeros(size(p1)),-ones(size(p1)),-p2.*t4,-p2.*t8,ones(size(p1)),zeros(size(p1)),t11,t12,-ones(size(p1)),...
        t17,t5,zeros(size(p1)),zeros(size(p1)),t11,-t5,zeros(size(p1)),zeros(size(p1)),t13,zeros(size(p1)),...
        t5,zeros(size(p1)),t12,zeros(size(p1)),-t5,zeros(size(p1)),t14,zeros(size(p1)),zeros(size(p1)),zeros(size(p1)),...
        -ones(size(p1)),zeros(size(p1)),zeros(size(p1)),zeros(size(p1)),ones(size(p1)),-p1.*t4,-p1.*t8,-ones(size(p1)),...
        t17,t13,t14,ones(size(p1)),zeros(size(p1))];
end
